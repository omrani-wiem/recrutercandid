"""
Seul point de sortie réel vers l'extérieur (effet de bord irréversible). Volontairement isolé
dans son propre serveur MCP pour que :

- aucun nœud LLM ne puisse l'appeler directement (il n'est jamais dans la liste d'outils
  passée aux appels client.messages.create() des nœuds d'extraction/explication/email)
- seul graph.py, APRÈS le passage par human_review_email_node ET vérification explicite de
  rejection_email_approved == True, ait le droit d'invoquer ce tool
- chaque envoi soit loggé de façon immuable (audit trail RH)
"""
from __future__ import annotations
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("email-server")

logging.basicConfig(filename="email_audit.log", level=logging.INFO)
logger = logging.getLogger("email_audit")

SMTP_HOST = "smtp.example-corp.com"
SMTP_PORT = 587
SENDER_ADDRESS = "recrutement@example-corp.com"


@mcp.tool()
def send_email(to_address: str, subject: str, body: str, approved_by: str) -> dict:
    """
    Envoie un e-mail. Cet outil DOIT uniquement être appelé après approbation humaine
    explicite — le paramètre approved_by est obligatoire et loggé pour audit.

    Args:
        to_address: adresse e-mail du destinataire
        subject: objet de l'e-mail
        body: corps du message (texte brut)
        approved_by: identifiant du recruteur ayant approuvé l'envoi (traçabilité)

    Returns:
        dict avec statut de l'envoi et timestamp
    """
    if not approved_by:
        raise ValueError("Envoi refusé : aucun approbateur humain identifié.")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_ADDRESS
    msg["To"] = to_address

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            # server.login(user, password)  # credentials via secret manager, pas en dur
            server.send_message(msg)
        status = "sent"
    except Exception as e:
        status = f"failed: {e}"

    logger.info(
        "EMAIL_SENT to=%s subject=%r approved_by=%s status=%s ts=%s",
        to_address, subject, approved_by, status, timestamp,
    )
    return {"status": status, "timestamp": timestamp, "approved_by": approved_by}


if __name__ == "__main__":
    mcp.run(transport="stdio")
