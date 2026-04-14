from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import string

from docx import Document
from pypdf import PdfReader

from prediction import analyze_email
from database import insert_analysis, fetch_stats, fetch_recent_activity, fetch_reports

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_text_fallback(save_path: str) -> str:
    with open(save_path, "rb") as f:
        raw = f.read()

    decoded = raw.decode("utf-8", errors="ignore").strip()
    if not decoded:
        decoded = raw.decode("latin-1", errors="ignore").strip()

    if not decoded:
        return ""

    printable_count = sum(ch in string.printable for ch in decoded)
    printable_ratio = printable_count / max(len(decoded), 1)
    return decoded if printable_ratio >= 0.6 else ""


def extract_uploaded_text(save_path: str, ext: str) -> str:
    if ext == "txt":
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == "pdf":
        try:
            reader = PdfReader(save_path, strict=False)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            parsed_text = "\n".join(pages_text).strip()
            if parsed_text:
                return parsed_text
        except Exception:
            pass

        fallback_text = _read_text_fallback(save_path)
        if fallback_text:
            return fallback_text
        raise RuntimeError("Unable to extract readable text from this PDF file.")

    if ext == "docx":
        doc = Document(save_path)
        return "\n".join(p.text for p in doc.paragraphs if p and p.text)

    raise RuntimeError("Unsupported file type.")


api_bp = Blueprint("api", __name__)
ui_bp = Blueprint("ui", __name__)


# ---------------- UI ROUTES ----------------


@ui_bp.route("/")
def dashboard_page():
    return render_template("dashboard.html")


@ui_bp.route("/email-analysis")
def email_analysis_page():
    return render_template("email_analysis.html")


@ui_bp.route("/reports")
def reports_page():
    return render_template("reports.html")


@ui_bp.route("/about")
def about_page():
    return render_template("about.html")


# ---------------- API ROUTES ----------------


@api_bp.route("/stats", methods=["GET"])
def api_stats():
    stats = fetch_stats()
    recent = fetch_recent_activity(limit=10)
    return jsonify({"stats": stats, "recent": recent})


@api_bp.route("/reports", methods=["GET"])
def api_reports():
    search = request.args.get("search", "").strip()
    risk_filter = request.args.get("risk_level", "").strip()
    prediction_filter = request.args.get("prediction", "").strip()

    reports = fetch_reports(
        search_term=search,
        risk_level=risk_filter,
        prediction=prediction_filter,
    )
    return jsonify({"reports": reports})


@api_bp.route("/analyze-email", methods=["POST"])
def api_analyze_email():
    subject = request.form.get("subject", "").strip()
    email_text = request.form.get("email_text", "").strip()
    file = request.files.get("file")

    if not email_text and file is None:
        return jsonify({"error": "Please provide email text or upload a file."}), 400

    # ---------- FILE UPLOAD ---------- #
    if file and file.filename:
        if not allowed_file(file.filename):
            return jsonify(
                {
                    "error": (
                        "Only .txt, .pdf, and .docx files are supported."
                    )
                }
            ), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        ext = filename.rsplit(".", 1)[1].lower()
        try:
            email_text = extract_uploaded_text(save_path=save_path, ext=ext)
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"Failed to process uploaded file: {exc}"}), 400

    email_text = email_text.strip()
    if not email_text:
        return jsonify({"error": "Email content cannot be empty."}), 400

    # ---------- RUN MODEL ---------- #
    try:
        analysis = analyze_email(email_text=email_text, subject=subject or "N/A")
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400

    # ---------- STORE RESULT (no email body stored) ---------- #
    insert_analysis(
        subject=analysis.get("subject", subject or "N/A"),
        prediction=analysis["prediction"],
        risk_level=analysis["risk_level"],
    )

    return jsonify(analysis)

