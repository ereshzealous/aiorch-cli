# Copyright 2026 Eresh Gorantla
# SPDX-License-Identifier: Apache-2.0

"""Kafka producer connector handler — publish-only for v1.

Implements the canonical ``publish`` operation against Kafka via
aiokafka. Per Q2 of the connectors design doc, every publish
starts a fresh AIOKafkaProducer, sends the message, and stops
the producer. No shared producer across operations.

Consumer / inbound trigger support is deliberately out of scope
for v1 — it requires a long-running subscription loop, consumer
group coordination, and offset management that don't fit the
stateless-handler model. Scoped into a separate future feature.

Supported protocols:
  PLAINTEXT                   — no auth, no TLS
  SSL                         — TLS only, no SASL
  SASL_PLAINTEXT              — SASL auth over plaintext
  SASL_SSL                    — SASL auth over TLS

Supported SASL mechanisms:
  PLAIN
  SCRAM-SHA-256
  SCRAM-SHA-512
"""

from __future__ import annotations

import logging
from typing import Any

from aiorch.connectors import ConnectorAuthError, NotSupportedError
from aiorch.connectors.handlers import ConnectorHandler, register_handler

logger = logging.getLogger("aiorch.connectors.handlers.kafka")


# ---------------------------------------------------------------------------
# Payload serialization
# ---------------------------------------------------------------------------
#
# Three explicit serializers plus a legacy 'auto' mode:
#
#   string  — accepts only str, encodes as UTF-8. Errors on dict/list/int/bool/bytes.
#   json    — accepts JSON-serializable values (dict, list, str, int, float, bool, None),
#             encodes via json.dumps. Errors on bytes (already-serialized).
#   bytes   — accepts bytes/bytearray only, passthrough. Errors on everything else.
#   auto    — legacy default that mirrors the pre-Phase-5 behavior: str→UTF-8,
#             dict/list→JSON, bytes→passthrough. Refuses int/bool/float instead
#             of falling into bytes(int)/bytes(bool) which produced silent garbage
#             (a 42-byte zero buffer for bytes(42), an empty buffer for bytes(True)).
#
# The 'auto' default is preserved so existing connectors keep working unchanged.
# New connectors should pick an explicit serializer. The auto option is ranked
# last in the UI dropdown and labelled "legacy" precisely so it rots out over
# time without breaking anyone on upgrade.

_VALUE_SERIALIZERS = ("auto", "json", "string", "bytes")


def _serialize_value(value: Any, serializer: str) -> bytes:
    """Encode a publish payload to bytes per the chosen serializer.

    Raises ValueError with a clear message on a type mismatch — no
    silent fallthrough to bytes() coercion that produces garbage.
    """
    if serializer not in _VALUE_SERIALIZERS:
        raise ValueError(
            f"Unknown value_serializer {serializer!r}. "
            f"Expected one of: {', '.join(_VALUE_SERIALIZERS)}."
        )

    if serializer == "string":
        if not isinstance(value, str):
            raise ValueError(
                f"value_serializer='string' requires a str payload, "
                f"got {type(value).__name__}. "
                f"Use 'json' for structured data or 'bytes' for raw binary."
            )
        return value.encode("utf-8")

    if serializer == "json":
        if isinstance(value, (bytes, bytearray)):
            raise ValueError(
                "value_serializer='json' refuses bytes input — the payload "
                "is already serialized. Use 'bytes' to send it verbatim."
            )
        import json as _json
        try:
            return _json.dumps(value).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"value_serializer='json' could not encode {type(value).__name__}: {exc}"
            ) from exc

    if serializer == "bytes":
        if not isinstance(value, (bytes, bytearray)):
            raise ValueError(
                f"value_serializer='bytes' requires bytes/bytearray, "
                f"got {type(value).__name__}. "
                f"Use 'string' for text or 'json' for structured data."
            )
        return bytes(value)

    # auto — legacy type-detect path, but with int/bool/float refused
    # explicitly instead of producing zero-length or garbage buffers.
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (dict, list)):
        import json as _json
        return _json.dumps(value).encode("utf-8")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError(
        f"value_serializer='auto' cannot infer encoding for {type(value).__name__}. "
        f"Pick an explicit value_serializer ('string', 'json', or 'bytes')."
    )


def _serialize_key(key: Any) -> bytes:
    """Encode a partition key to bytes.

    Keys are almost always strings (entity IDs, hash buckets); we accept
    str + bytes/bytearray and refuse everything else with a clear message.
    Callers passing integer IDs should str() them at the call site so
    partitioning is deterministic and matches what other producers see.
    """
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    raise ValueError(
        f"Kafka 'key' must be a string or bytes, got {type(key).__name__}. "
        f"For integer IDs pass str(value) at the pipeline step so partition "
        f"hashing is deterministic across producers."
    )


# ---------------------------------------------------------------------------
# Producer properties — KV escape hatch for any aiokafka kwarg not exposed
# as a typed UI field. Phase 8.
# ---------------------------------------------------------------------------
#
# We expose ~10 producer kwargs as typed UI fields. aiokafka supports
# several dozen more (linger_ms, batch_size, retries, transactional_id,
# etc.). Rather than adding a UI field per knob, the connector accepts
# a `producer_properties` dict from the form and merges it into the
# producer kwargs at start time.
#
# Type coercion: aiokafka expects ints for ms/byte fields and bools for
# flag fields, but the KV editor stores strings. We maintain a small
# type map for well-known kwargs and pass everything else through. If
# a user sets an unknown kwarg with a wrong type, aiokafka raises a
# clean error naming the offending key.
#
# Collision policy: if a KV row's name matches a kwarg the typed UI
# already sets (acks, compression_type, etc.), the typed value wins
# and we log a warning. The KV editor is for the *long tail*, not for
# overriding things visible in the form.

# Names the typed UI fields populate. KV rows with these keys are
# ignored (with a warning) so the form is always the source of truth
# for things users see.
_TYPED_PRODUCER_KWARGS = {
    "bootstrap_servers",
    "client_id",
    "acks",
    "compression_type",
    "enable_idempotence",
    "max_in_flight_requests_per_connection",
    "request_timeout_ms",
    "security_protocol",
    "sasl_mechanism",
    "sasl_plain_username",
    "sasl_plain_password",
    "ssl_context",
}


def _parse_kafka_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off", ""):
        return False
    raise ValueError(f"Cannot parse {value!r} as bool")


# Best-effort type coercion for well-known aiokafka producer kwargs.
# Unknown keys pass through as strings — aiokafka raises on a real
# type mismatch with a clear error.
_PRODUCER_KWARG_TYPES: dict[str, Any] = {
    "linger_ms": int,
    "batch_size": int,
    "max_request_size": int,
    "max_in_flight_requests_per_connection": int,
    "retries": int,
    "retry_backoff_ms": int,
    "delivery_timeout_ms": int,
    "metadata_max_age_ms": int,
    "connections_max_idle_ms": int,
    "reconnect_backoff_ms": int,
    "reconnect_backoff_max_ms": int,
    "transaction_timeout_ms": int,
    "send_backoff_ms": int,
    "buffer_memory": int,
    "enable_idempotence": _parse_kafka_bool,
    "transactional_id": str,
    "client_id": str,
}


def _normalize_kafka_property_name(name: str) -> str:
    """Accept both Java Kafka dot-case (`linger.ms`) and aiokafka
    snake_case (`linger_ms`). Most users copy-paste from the upstream
    Kafka docs so we normalize transparently."""
    return name.replace(".", "_").strip()


def _apply_producer_properties(producer_kwargs: dict, props: Any) -> None:
    """Merge UI KV producer properties into the producer kwargs dict.

    Mutates `producer_kwargs` in place. Raises ValueError on a
    well-known property whose value can't be coerced to its expected
    type — better to fail at dispatch than at AIOKafkaProducer.start().
    """
    if not props:
        return
    if not isinstance(props, dict):
        raise ValueError(
            f"producer_properties must be a dict, got {type(props).__name__}"
        )
    for raw_name, raw_value in props.items():
        name = _normalize_kafka_property_name(str(raw_name))
        if not name:
            continue
        if name in _TYPED_PRODUCER_KWARGS:
            logger.warning(
                "Kafka producer_properties: ignoring %r (=%r) — "
                "overridden by a typed UI field. Set it via the form.",
                name, raw_value,
            )
            continue
        coerce = _PRODUCER_KWARG_TYPES.get(name)
        if coerce is not None:
            try:
                producer_kwargs[name] = coerce(raw_value)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Kafka producer_properties: {name}={raw_value!r} "
                    f"could not be coerced to {getattr(coerce, '__name__', str(coerce))}: {exc}"
                ) from exc
        else:
            # Unknown kwarg — pass through verbatim. aiokafka will
            # raise a TypeError with the kwarg name if the type is wrong.
            producer_kwargs[name] = raw_value


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _map_kafka_error(exc: Exception) -> Exception:
    """Translate aiokafka errors into friendly messages.

    aiokafka has churned on error class names across versions. We
    import each one defensively and skip the ones that don't exist
    in the installed version.
    """
    try:
        import aiokafka.errors as kerr  # type: ignore
    except ImportError:
        return exc

    def _cls(*names: str) -> tuple:
        """Pick whichever of the given class names exists in this aiokafka."""
        classes = tuple(getattr(kerr, n) for n in names if hasattr(kerr, n))
        return classes or (type(None),)

    auth_classes = _cls(
        "AuthenticationFailedError",
        "SASLAuthenticationFailedError",
        "IllegalSaslStateError",
    )
    topic_auth_classes = _cls("TopicAuthorizationFailedError")
    unknown_topic_classes = _cls("UnknownTopicOrPartitionError")
    invalid_topic_classes = _cls("InvalidTopicError")
    connection_classes = _cls("KafkaConnectionError", "NoBrokersAvailable")

    if isinstance(exc, auth_classes):
        return ConnectorAuthError(
            f"Kafka authentication failed: {exc}. "
            f"Check sasl_username and sasl_password on the connector."
        )
    if isinstance(exc, topic_auth_classes):
        return PermissionError(
            f"Kafka topic authorization denied: {exc}. "
            f"The connector's credentials don't have write access to this topic."
        )
    if isinstance(exc, unknown_topic_classes):
        return ValueError(
            f"Kafka topic does not exist: {exc}. "
            f"Check the 'destination' value or create the topic first."
        )
    if isinstance(exc, invalid_topic_classes):
        return ValueError(f"Invalid Kafka topic name: {exc}")
    if isinstance(exc, connection_classes):
        return ConnectionError(
            f"Cannot reach Kafka bootstrap servers: {exc}. "
            f"Check bootstrap_servers config and network access."
        )

    return exc


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class KafkaHandler(ConnectorHandler):
    """Producer-only Kafka handler via aiokafka."""

    type = "stream"
    subtype = "kafka"
    supported_operations = {"publish"}

    @classmethod
    def get_client_info(cls) -> dict | None:
        try:
            import aiokafka  # type: ignore
            version = getattr(aiokafka, "__version__", "unknown")
        except ImportError:
            return None
        return {
            "client_library": "aiokafka",
            "client_version": version,
            "docs_url": "https://kafka.apache.org/documentation/#producerconfigs",
        }

    async def execute(
        self,
        *,
        operation: str,
        params: dict,
        connector_config: dict,
        context: dict,
    ) -> Any:
        self._check_supported(operation)

        if operation != "publish":
            raise NotSupportedError(
                f"KafkaHandler v1 supports only 'publish'. "
                f"Consumer support is a future feature."
            )

        try:
            return await self._publish(params, connector_config)
        except Exception as exc:
            mapped = _map_kafka_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc

    async def _publish(self, params: dict, connector_config: dict) -> dict:
        destination = params.get("destination") or params.get("topic")
        payload = params.get("payload") or params.get("value")
        metadata = params.get("metadata") or {}
        key = params.get("key")

        if not destination:
            raise ValueError(
                "publish operation requires 'destination' (Kafka topic name)"
            )
        if payload is None:
            raise ValueError(
                "publish operation requires 'payload' (bytes or string)"
            )

        # Per-step override wins over the connector-level default;
        # connector default falls back to legacy 'auto' so existing
        # connectors keep working unchanged.
        value_serializer = (
            params.get("value_serializer")
            or connector_config.get("value_serializer")
            or "auto"
        )
        payload_bytes = _serialize_value(payload, value_serializer)

        # Optional partition key as bytes
        key_bytes: bytes | None = None
        if key is not None:
            key_bytes = _serialize_key(key)

        # Metadata becomes Kafka message headers: list[tuple[str, bytes]].
        # Same bytes-coercion trap as the payload path — bytes(int) in
        # Python 3 produces a zero-filled buffer of that length, bytes(bool)
        # produces empty bytes. Refuse non-string/bytes values loudly
        # so "traceparent: 42" doesn't silently land as b"\x00" * 42 on
        # the wire. Callers that need numeric headers should str(value)
        # at the step level.
        headers: list[tuple[str, bytes]] = []
        if isinstance(metadata, dict):
            for k, v in metadata.items():
                if v is None:
                    continue
                if isinstance(v, str):
                    v_bytes = v.encode("utf-8")
                elif isinstance(v, (bytes, bytearray)):
                    v_bytes = bytes(v)
                else:
                    raise ValueError(
                        f"Kafka header {k!r} must be a string or bytes, "
                        f"got {type(v).__name__}. For numeric or boolean "
                        f"values, stringify them at the pipeline step "
                        f"(e.g. `metadata: {{{k}: '{{{{value}}}}'}}`)."
                    )
                headers.append((str(k), v_bytes))

        # Build the producer
        from aiokafka import AIOKafkaProducer

        producer_kwargs: dict = {
            "bootstrap_servers": connector_config.get("bootstrap_servers"),
            "client_id": connector_config.get("client_id", "aiorch-connector"),
            "acks": connector_config.get("acks", "all"),
        }

        compression = connector_config.get("compression_type")
        if compression and compression != "none":
            producer_kwargs["compression_type"] = compression

        # Idempotent producer: exactly-once per-partition semantics.
        # Forces acks=all and max_in_flight=1, which is required by
        # the aiokafka contract; we apply the overrides so operators
        # enabling this don't also have to remember the coupling.
        if bool(connector_config.get("enable_idempotence", False)):
            producer_kwargs["enable_idempotence"] = True
            producer_kwargs["acks"] = "all"
            producer_kwargs["max_in_flight_requests_per_connection"] = 1

        request_timeout_ms = connector_config.get("request_timeout_ms")
        if request_timeout_ms is not None:
            producer_kwargs["request_timeout_ms"] = int(request_timeout_ms)

        security_protocol = connector_config.get("security_protocol", "PLAINTEXT")
        producer_kwargs["security_protocol"] = security_protocol

        if security_protocol in ("SASL_PLAINTEXT", "SASL_SSL"):
            producer_kwargs["sasl_mechanism"] = connector_config.get(
                "sasl_mechanism", "PLAIN",
            )
            producer_kwargs["sasl_plain_username"] = connector_config.get("sasl_username")
            producer_kwargs["sasl_plain_password"] = connector_config.get("sasl_password")

        if security_protocol in ("SSL", "SASL_SSL"):
            ssl_ctx = connector_config.get("ssl_context")
            if ssl_ctx is None:
                # Build an SSLContext from PEM-string config fields so
                # operators can configure mTLS / Confluent Cloud / MSK
                # entirely from the UI without managing files on disk.
                ssl_cafile = connector_config.get("ssl_cafile")
                ssl_certfile = connector_config.get("ssl_certfile")
                ssl_keyfile = connector_config.get("ssl_keyfile")
                ssl_password = connector_config.get("ssl_password")

                if ssl_cafile or ssl_certfile or ssl_keyfile:
                    import os
                    import ssl as _ssl
                    import tempfile

                    ctx = _ssl.create_default_context()

                    if ssl_cafile:
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".pem", delete=False
                        ) as f:
                            f.write(ssl_cafile)
                            ca_path = f.name
                        try:
                            ctx.load_verify_locations(ca_path)
                        finally:
                            os.unlink(ca_path)

                    if ssl_certfile and ssl_keyfile:
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".pem", delete=False
                        ) as cf, tempfile.NamedTemporaryFile(
                            mode="w", suffix=".pem", delete=False
                        ) as kf:
                            cf.write(ssl_certfile)
                            kf.write(ssl_keyfile)
                            cert_path, key_path = cf.name, kf.name
                        try:
                            ctx.load_cert_chain(
                                cert_path, key_path, password=ssl_password or None,
                            )
                        finally:
                            os.unlink(cert_path)
                            os.unlink(key_path)

                    ssl_ctx = ctx

            if ssl_ctx is not None:
                producer_kwargs["ssl_context"] = ssl_ctx

        # Apply user-supplied producer properties LAST so the typed
        # collision warning fires after the typed fields are already set.
        # The function refuses to overwrite typed kwargs (warns instead),
        # so this is safe — typed UI fields are always authoritative.
        _apply_producer_properties(
            producer_kwargs,
            connector_config.get("producer_properties"),
        )

        if not producer_kwargs["bootstrap_servers"]:
            raise ValueError(
                "Kafka connector requires 'bootstrap_servers' in config"
            )

        producer = AIOKafkaProducer(**producer_kwargs)
        await producer.start()
        try:
            record_metadata = await producer.send_and_wait(
                destination,
                value=payload_bytes,
                key=key_bytes,
                headers=headers if headers else None,
            )
            return {
                "id": f"{record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}",
                "topic": record_metadata.topic,
                "partition": record_metadata.partition,
                "offset": record_metadata.offset,
                "timestamp": record_metadata.timestamp,
                "bytes_transferred": len(payload_bytes),
            }
        finally:
            try:
                await producer.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Consumer placeholder — Phase 8
# ---------------------------------------------------------------------------
#
# The kafka-consumer subtype is registered so the dispatcher can refuse
# attempts to use it with a clear error message. Pipelines that say
# `subtype: kafka-consumer` get a NotSupportedError at dispatch instead
# of a confusing "no handler registered" lookup failure. The UI shows
# the subtype as "Coming soon" and disables the Save button.


class KafkaConsumerPlaceholder(ConnectorHandler):
    """Reserved subtype — consumer support is planned for v1.1."""

    type = "stream"
    subtype = "kafka-consumer"
    supported_operations = {"consume", "subscribe", "poll"}

    async def execute(self, *, operation: str, params, connector_config, context):
        raise NotSupportedError(
            "Kafka consumer connector is not yet implemented. "
            "Producer support ships in v1; consumer needs subscription loops, "
            "consumer group coordination, and offset management that don't fit "
            "the stateless handler model. Targeted for v1.1. "
            "Use a webhook trigger or scheduled pull pipeline as a workaround."
        )


# Register on import. The producer handler is registered under both
# the new canonical name `kafka-producer` and the legacy `kafka` name
# so existing connectors with `subtype=kafka` in the database resolve
# unchanged. New connectors created via the UI use kafka-producer.

_producer = KafkaHandler()
register_handler(_producer)  # legacy ("stream", "kafka")

_producer_aliased = KafkaHandler()
_producer_aliased.subtype = "kafka-producer"
register_handler(_producer_aliased)  # canonical ("stream", "kafka-producer")

register_handler(KafkaConsumerPlaceholder())
