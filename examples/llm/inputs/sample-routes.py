"""Minimal Flask-style routes for the OpenAPI extraction demo."""

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/api/projects", methods=["GET"])
def list_projects():
    """List all projects. Supports optional ?status=active filter."""
    status = request.args.get("status")
    # ... fetch from DB ...
    return jsonify([])


@app.route("/api/projects/<int:project_id>", methods=["GET"])
def get_project(project_id: int):
    """Get a single project by id."""
    # ... fetch from DB ...
    return jsonify({"id": project_id})


@app.route("/api/projects", methods=["POST"])
def create_project():
    """Create a new project. Body: {name, owner, budget}."""
    body = request.get_json()
    # ... insert into DB ...
    return jsonify({"id": 42, "name": body["name"]}), 201


@app.route("/api/projects/<int:project_id>", methods=["PATCH"])
def update_project(project_id: int):
    """Partially update a project. Body: any subset of {name, owner, budget, status}."""
    body = request.get_json()
    # ... update in DB ...
    return jsonify({"id": project_id, **body})


@app.route("/api/projects/<int:project_id>", methods=["DELETE"])
def delete_project(project_id: int):
    """Soft-delete a project (is_active=false)."""
    # ... mark inactive ...
    return "", 204


@app.route("/api/projects/<int:project_id>/runs", methods=["GET"])
def list_project_runs(project_id: int):
    """List runs for a project, most recent first. Supports ?limit=N (default 50)."""
    limit = int(request.args.get("limit", 50))
    # ... fetch from DB ...
    return jsonify([])


@app.route("/api/health", methods=["GET"])
def health():
    """Liveness probe. Returns 200 with basic status when the app is serving."""
    return jsonify({"status": "ok"})
