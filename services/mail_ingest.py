import email
import imaplib
import os
from email.header import decode_header, make_header

MAX_PDF_BYTES = 10 * 1024 * 1024


def fetch_cvs_from_mailbox(max_messages: int = 20) -> list[dict]:
    """Récupère les PDF des mails non lus. Lire un mail le marque « lu » : pas de doublon."""
    host = os.getenv("IMAP_HOST", "imap.gmail.com")
    user, password = os.getenv("IMAP_USER"), os.getenv("IMAP_PASSWORD")
    if not user or not password:
        raise RuntimeError("IMAP_USER / IMAP_PASSWORD manquants dans .env")

    found = []
    with imaplib.IMAP4_SSL(host) as imap:
        imap.login(user, password)
        imap.select("INBOX")
        _, data = imap.search(None, "UNSEEN")
        for num in data[0].split()[:max_messages]:
            _, msg_data = imap.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            sender = email.utils.parseaddr(msg.get("From", ""))[1]
            for part in msg.walk():
                raw_name = part.get_filename()
                if not raw_name:
                    continue
                filename = str(make_header(decode_header(raw_name)))
                if not filename.lower().endswith(".pdf"):
                    continue
                content = part.get_payload(decode=True)
                if content and content[:4] == b"%PDF" and len(content) <= MAX_PDF_BYTES:
                    found.append({"filename": filename, "content": content, "sender": sender})
    return found