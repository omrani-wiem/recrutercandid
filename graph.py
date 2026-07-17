"""
Assemble tous les nœuds en un graphe LangGraph. Deux points d'interruption garantissent
qu'aucune décision de rejet finale ni aucun envoi d'e-mail ne se produit sans validation
humaine explicite.
"""
from __future__ import annotations
import argparse
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from state import PipelineState
from nodes.ocr import ocr_node, needs_vlm_fallback
from nodes.extraction import extraction_node
from nodes.jd_parsing import jd_parsing_node
from nodes.skill_relevance import skill_relevance_node
from nodes.matching_scoring import matching_scoring_node, routing_decision
from nodes.guardrails import guardrails_node
from nodes.explanation import explanation_node
from nodes.rejection_email import rejection_email_node
from nodes.human_review import human_review_decision_node, human_review_email_node


def _build_resend_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "cv-matching-system/1.0",
    }


def qwen_vlm_ocr_node(state: dict) -> dict:
    """
    Appelle le serveur MCP OCR (client MCP standard, transport stdio).

    Ce nœud n'est déclenché QUE si l'extraction native a une confiance trop basse
    (needs_vlm_fallback). Il suppose qu'un serveur d'inférence Qwen2.5-VL tourne déjà
    (voir mcp_servers/ocr_mcp_server.py et la variable QWEN_VLM_URL). Avant le correctif,
    toute erreur ici (serveur non démarré, timeout réseau, etc.) faisait planter tout le
    pipeline avec une exception non gérée. Maintenant on capture l'échec et on le
    remonte proprement dans l'état, pour que l'UI puisse afficher un message clair
    plutôt qu'un traceback.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _call():
        params = StdioServerParameters(command="python", args=["mcp_servers/ocr_mcp_server.py"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("extract_from_image", {"pdf_path": state["cv_path"]})
                return result

    import asyncio
    try:
        result = asyncio.run(_call())
    except Exception as exc:
        return {
            "extracted_text": "",
            "ocr_confidence": 0.0,
            "ocr_method": "qwen_vlm_failed",
            "ocr_error": (
                "L'OCR de secours (Qwen VLM) a échoué : "
                f"{exc}. Vérifiez que le serveur d'inférence est bien démarré "
                "(voir mcp_servers/ocr_mcp_server.py) et que QWEN_VLM_URL est correct."
            ),
        }

    return {
        "extracted_text": result["text"],
        "ocr_confidence": result["confidence"],
        "ocr_method": "qwen_vlm",
    }


def send_rejection_email(state: dict) -> dict:
    """
    Appelé uniquement si rejection_email_approved == True. Le nœud lui-même revérifie la
    condition en dur (défense en profondeur, ne fait jamais confiance à un seul contrôle).
    """
    import asyncio
    import json
    import os
    import urllib.error
    import urllib.request
    from dotenv import load_dotenv

    load_dotenv()
    override = state.get("rejection_email_override") or {}
    if not state.get("rejection_email_approved") and not override:
        return {"rejection_email_sent": False, "email_send_error": "Envoi non approuvé"}

    resend_url = os.getenv("RESEND_MCP_URL", "https://mcp.resend.com/mcp")
    resend_api_key = os.getenv("RESEND_API_KEY")
    draft = state.get("rejection_email_draft") or {}
    to_address = override.get("to") or getattr(state.get("resume"), "email", None) or ""
    sender_address = override.get("from") or os.getenv("RESEND_SENDER_EMAIL", "onboarding@resend.dev")
    subject = override.get("subject") or draft.get("subject", "Suite à votre candidature")
    body = override.get("body") or draft.get("body", "")

    if not resend_api_key:
        return {"rejection_email_sent": False, "email_send_error": "RESEND_API_KEY manquante"}
    if not to_address:
        return {"rejection_email_sent": False, "email_send_error": "Aucune adresse e-mail de destinataire"}
    if not sender_address:
        return {"rejection_email_sent": False, "email_send_error": "Aucune adresse expéditrice Resend configurée"}

    async def _call_mcp():
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        headers = _build_resend_headers(resend_api_key)
        async with sse_client(url=resend_url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("send_email", {
                    "to": to_address,
                    "subject": subject,
                    "text": body,
                    "from": sender_address,
                })
                return result

    def _call_rest() -> dict:
        payload = {
            "from": sender_address,
            "to": [to_address],
            "subject": subject,
            "text": body,
        }
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=json.dumps(payload).encode("utf-8"),
            headers=_build_resend_headers(resend_api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                return {"rejection_email_sent": response.status in (200, 202), "email_send_error": None if response.status in (200, 202) else f"Resend API returned {response.status}"}
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="ignore")
            return {"rejection_email_sent": False, "email_send_error": f"Resend API HTTP {exc.code}: {body_text}"}
        except Exception as exc:
            return {"rejection_email_sent": False, "email_send_error": f"Resend API request failed: {exc}"}

    try:
        result = asyncio.run(_call_mcp())
        return {"rejection_email_sent": bool(result), "email_send_error": None if bool(result) else "Réponse vide du serveur MCP"}
    except Exception as exc:
        rest_result = _call_rest()
        if rest_result.get("rejection_email_sent"):
            return rest_result
        return {"rejection_email_sent": False, "email_send_error": f"Échec d'envoi MCP: {exc} | {rest_result.get('email_send_error', 'fallback REST failed')}"}




def route_after_email_review(state: dict) -> str:
    return "send_email" if state.get("rejection_email_approved") else END


def route_after_rejection_email(state: dict) -> str:
    """
    Route toujours vers la revue humaine après génération du brouillon, SAUF si
    l'envoi a déjà été explicitement approuvé par un humain au préalable (ce qui
    n'arrive normalement jamais à ce stade précis du graphe, mais on garde le check
    par cohérence si ce nœud est un jour ré-atteint après une reprise).

    Avant ce correctif, la simple présence d'un email candidat suffisait à envoyer
    le mail directement, sans jamais passer par human_review_email_node — ce qui
    revenait à rendre la revue humaine obligatoire (voir nodes/human_review.py)
    inopérante dans la quasi-totalité des cas réels (la plupart des CV contiennent
    un email).
    """
    if state.get("rejection_email_approved"):
        return "send_email"
    return "human_review_email"


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("ocr", ocr_node)
    graph.add_node("qwen_vlm_ocr", qwen_vlm_ocr_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("jd_parsing", jd_parsing_node)
    graph.add_node("matching_scoring", matching_scoring_node)
    graph.add_node("skill_relevance", skill_relevance_node)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("human_review_decision", human_review_decision_node)
    graph.add_node("rejection_email", rejection_email_node)
    graph.add_node("human_review_email", human_review_email_node)
    graph.add_node("send_email", send_rejection_email)

    graph.add_edge(START, "ocr")

    # OCR : branchement natif vs fallback Qwen VLM
    graph.add_conditional_edges("ocr", needs_vlm_fallback, {
        "qwen_vlm_ocr": "qwen_vlm_ocr",
        "extraction": "extraction",
    })
    graph.add_edge("qwen_vlm_ocr", "extraction")

    # parsing JD en parallèle logique, convergent vers le scoring
    # jd_parsing ne dépend pas du résultat de l'extraction (seulement de jd_text, présent
    # dès le départ) : on l'enchaîne quand même après extraction pour éviter tout fan-in
    # (voir note plus haut sur le bug de double-exécution).
    graph.add_edge("extraction", "jd_parsing")
    graph.add_edge("jd_parsing", "skill_relevance")
    graph.add_edge("skill_relevance", "matching_scoring")

    graph.add_edge("matching_scoring", "guardrails")
    graph.add_edge("guardrails", "explanation")

    # routage post-scoring : shortlist direct / reject direct / zone grise -> humain
    graph.add_conditional_edges("explanation", routing_decision, {
        "shortlist": END,
        "reject": "rejection_email",
        "needs_human_review": "human_review_decision",
    })

    graph.add_conditional_edges(
        "human_review_decision",
        lambda s: "rejection_email" if s["decision"] == "reject" else END,
        {"rejection_email": "rejection_email", END: END},
    )

    graph.add_conditional_edges("rejection_email", route_after_rejection_email, {
        "send_email": "send_email",
        "human_review_email": "human_review_email",
    })
    graph.add_conditional_edges("human_review_email", route_after_email_review, {
        "send_email": "send_email", END: END,
    })
    graph.add_edge("send_email", END)

    # checkpointer obligatoire : interrupt() a besoin d'un état persistant pour reprendre
    # NB : sans ceci, LangGraph refuse de (dé)sérialiser nos modèles Pydantic
    # (ResumeProfile, JobRequirements, MatchScore, SecurityFlag) dans le checkpoint,
    # et la reprise après interrupt() rejoue la même pause au lieu de continuer
    # au nœud suivant.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("state", "ResumeProfile"),
        ("state", "JobRequirements"),
        ("state", "MatchScore"),
        ("state", "SecurityFlag"),
    ])
    return graph.compile(checkpointer=MemorySaver(serde=serde))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", required=True)
    parser.add_argument("--jd", required=True)
    args = parser.parse_args()

    app = build_graph()
    config = {"configurable": {"thread_id": "demo-run-1"}}

    with open(args.jd) as f:
        jd_text = f.read()

    result = app.invoke({"cv_path": args.cv, "jd_text": jd_text}, config=config)
    print(result)

    # si le graphe s'est arrêté sur une interruption, reprendre avec :
    # app.invoke(Command(resume={"decision": "reject"}), config=config)
    # app.invoke(Command(resume={"action": "approve"}), config=config)
