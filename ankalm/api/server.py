"""Localhost-only FastAPI server providing OpenAI-compatible Chat, Multimodal Document/Excel/Image/Audio Intelligence, Hybrid RAG, Tools, MoE, and Training Control Center (Section 17 & 18)."""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ankalm.chat.database import ChatDatabase
from ankalm.chat.hybrid_rag import HybridRAGEngine
from ankalm.core.config import AnkaLMConfig, ModelProfileConfig
from ankalm.core.constants import CHECKPOINT_MATURITY_LEVELS, PROJECT_ROOT
from ankalm.evaluation.suite import run_turkish_evaluation_lab
from ankalm.hardware.detector import discover_hardware
from ankalm.inference.engine import AnkaLMInferenceEngine, GenerationConfig
from ankalm.inference.speculative import SpeculativeDecoder
from ankalm.multimodal.pipeline import AnkaLMDocumentIntelligencePipeline
from ankalm.storage.checkpoint import CheckpointManager
from ankalm.storage.manager import StorageManager
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer
from ankalm.tools.registry import ToolCallRequest, get_default_tool_registry
from ankalm.training.process_manager import TrainingProcessManager
from ankalm.utils.security import validate_safe_path, verify_localhost_binding


class ChatMessageInput(BaseModel):
    role: str = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "latest"
    messages: List[ChatMessageInput]
    stream: bool = False
    max_tokens: int = 96
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    greedy: bool = False
    repetition_penalty: float = 1.15
    seed: Optional[int] = 42
    conversation_id: Optional[str] = None
    use_rag: bool = False
    workspace: Optional[str] = None
    speculative_decoding: bool = False


class EmbeddingsRequest(BaseModel):
    input: str | List[str]
    model: str = "AnkaLM-BiEncoder-Embed-v1"


class TokenizeRequest(BaseModel):
    text: str
    add_bos: bool = False
    add_eos: bool = False


class DetokenizeRequest(BaseModel):
    token_ids: List[int]
    skip_special_tokens: bool = True


class ModelLoadRequest(BaseModel):
    checkpoint: str = "latest"
    precision: str = "fp32"


class ConversationCreateRequest(BaseModel):
    title: str = "Yeni Türkçe Sohbet"
    system_prompt: str = "Sen AnkaLM, yerel donanımda sıfırdan eğitilmiş Türkçe yapay zeka asistanısın."
    model_checkpoint: str = "latest"


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    model_checkpoint: Optional[str] = None


class MessageEditRequest(BaseModel):
    content: str
    truncate_after: bool = True


class TrainingStartRequest(BaseModel):
    config_path: str = "configs/small.yaml"
    resume_from: Optional[str] = None
    max_steps: Optional[int] = None


class DocumentQueryRequest(BaseModel):
    question: str
    filename: Optional[str] = None
    workspace: Optional[str] = None
    top_k: int = 4


class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    user_authorized: bool = False
    timeout_sec: Optional[float] = None


class RuntimeState:
    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.ckpt_manager = CheckpointManager(root / "checkpoints")
        self.proc_manager = TrainingProcessManager(root / "logs")
        self.storage_manager = StorageManager(root=root)
        self.chat_db = ChatDatabase(root / "cache" / "conversations.sqlite3")
        self.hybrid_rag = HybridRAGEngine(root / "cache" / "rag")
        self.doc_pipeline = AnkaLMDocumentIntelligencePipeline(self.hybrid_rag)
        self.tool_registry = get_default_tool_registry()
        self.gen_config = GenerationConfig()
        self.loaded_engine: Optional[AnkaLMInferenceEngine] = None
        self.loaded_checkpoint_name: Optional[str] = None

    def get_tokenizer(self) -> AnkaLMTokenizer:
        if self.loaded_engine is not None:
            return self.loaded_engine.tokenizer
        return AnkaLMTokenizer.load(self.root / "tokenizer")

    def get_or_load_engine(self, checkpoint_name: str = "latest", precision: str = "fp32") -> AnkaLMInferenceEngine:
        if (
            self.loaded_engine is not None
            and self.loaded_checkpoint_name == checkpoint_name
            and self.loaded_engine.precision == precision.lower()
        ):
            return self.loaded_engine

        target = checkpoint_name
        if target in ("auto", "latest"):
            for candidate in ("small_sft", "small_final", "moe_final", "edge_final", "latest", "best", "tiny_final", "micro_final"):
                try:
                    p = self.ckpt_manager.resolve_checkpoint_dir(candidate)
                    with open(p / "metadata.json", "r", encoding="utf-8") as f:
                        m = json.load(f)
                    if m.get("maturity_level") not in ("random_init", "architecture_config_only") and m.get("global_step", 0) > 0:
                        target = candidate
                        break
                except Exception:
                    continue

        engine = AnkaLMInferenceEngine.from_checkpoint(
            target,
            checkpoints_dir=self.root / "checkpoints",
            tokenizer_dir=self.root / "tokenizer",
            precision=precision,
            allow_random_init=False,
        )
        self.loaded_engine = engine
        self.loaded_checkpoint_name = target
        return engine


def create_app(root: Path = PROJECT_ROOT) -> FastAPI:
    app = FastAPI(
        title="AnkaLM AI Studio — Multimodal Local Turkish AI Ecosystem & API",
        version="0.2.0",
        description="100% Local Turkish Foundation Model Family (Dense, MoE, Edge), PDF/Excel/Image/OCR/Vision Studio, Hybrid RAG & Tool Sandbox (127.0.0.1 only)",
    )
    state = RuntimeState(root=root)
    web_dir = root / "web"

    @app.get("/health")
    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        ckpts = state.ckpt_manager.list_checkpoints()
        trained_ckpts = [
            c for c in ckpts
            if c.maturity_level not in ("random_init", "architecture_config_only") and c.global_step > 0
        ]
        return {
            "status": "ok",
            "local_only": True,
            "host": "127.0.0.1",
            "inference_ready": len(trained_ckpts) > 0,
            "loaded_model": state.loaded_engine.metadata.model_name if state.loaded_engine else None,
            "loaded_checkpoint": state.loaded_checkpoint_name,
            "loaded_maturity": state.loaded_engine.metadata.maturity_level if state.loaded_engine else None,
            "available_trained_checkpoints": len(trained_ckpts),
            "total_checkpoints": len(ckpts),
            "multimodal_engines": ["pdf", "excel", "image_ocr_vision", "docx", "pptx", "html", "wav_audio", "hybrid_rag", "tool_sandbox"],
        }

    @app.get("/models")
    @app.get("/v1/models")
    @app.get("/api/models")
    def list_models() -> Dict[str, Any]:
        ckpts = state.ckpt_manager.list_checkpoints()
        data = []
        for c in ckpts:
            is_ready = (c.maturity_level not in ("random_init", "architecture_config_only")) and (c.global_step > 0) and c.weights_updated_verified
            m_cfg = ModelProfileConfig(**c.model_config)
            tot_p, act_p = m_cfg.estimate_total_and_active_parameters()
            data.append({
                "id": c.checkpoint_name,
                "object": "model",
                "created": c.created_at,
                "owned_by": "ankalm-local",
                "model_name": c.model_name,
                "profile": c.model_profile,
                "model_family": m_cfg.model_family,
                "use_moe": m_cfg.use_moe,
                "num_experts": m_cfg.num_experts,
                "top_k_experts": m_cfg.top_k_experts,
                "parameter_count": c.parameter_count,
                "total_stored_parameters": tot_p,
                "active_parameters_per_token": act_p,
                "global_step": c.global_step,
                "processed_tokens": c.processed_tokens,
                "train_loss": c.train_loss,
                "val_loss": c.val_loss,
                "val_perplexity": c.val_perplexity,
                "maturity_level": c.maturity_level,
                "maturity_description": CHECKPOINT_MATURITY_LEVELS.get(c.maturity_level, c.maturity_level),
                "inference_ready": is_ready,
                "weights_updated_verified": c.weights_updated_verified,
                "param_l2_delta_from_init": c.param_l2_delta_from_init,
            })
        return {"object": "list", "data": data}

    @app.get("/model/info")
    def get_model_info_query(checkpoint: str = Query("latest")) -> Dict[str, Any]:
        return get_model_info(checkpoint)

    @app.get("/api/models/{model_id}")
    def get_model_info(model_id: str) -> Dict[str, Any]:
        ckpt_dir = state.ckpt_manager.resolve_checkpoint_dir(model_id)
        with open(ckpt_dir / "metadata.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        m_cfg = ModelProfileConfig(**d.get("model_config", {}))
        tot_p, act_p = m_cfg.estimate_total_and_active_parameters()
        d["total_stored_parameters"] = tot_p
        d["active_parameters_per_token"] = act_p
        d["inference_ready"] = (
            d.get("maturity_level") not in ("random_init", "architecture_config_only")
            and d.get("global_step", 0) > 0
            and d.get("weights_updated_verified", False)
        )
        return d

    @app.post("/api/models/load")
    def load_model(req: ModelLoadRequest) -> Dict[str, Any]:
        try:
            eng = state.get_or_load_engine(req.checkpoint, precision=req.precision)
            tot_p, act_p = eng.model.config.estimate_total_and_active_parameters()
            return {
                "status": "loaded",
                "checkpoint": eng.metadata.checkpoint_name,
                "model_name": eng.metadata.model_name,
                "maturity_level": eng.metadata.maturity_level,
                "maturity_description": eng.metadata.maturity_description,
                "precision": eng.precision,
                "parameter_count": eng.metadata.parameter_count,
                "total_stored_parameters": tot_p,
                "active_parameters_per_token": act_p,
                "global_step": eng.metadata.global_step,
            }
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/models/unload")
    def unload_model() -> Dict[str, Any]:
        state.loaded_engine = None
        state.loaded_checkpoint_name = None
        return {"status": "unloaded"}

    @app.post("/embeddings")
    @app.post("/v1/embeddings")
    def create_embeddings(req: EmbeddingsRequest) -> Dict[str, Any]:
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        data = []
        for idx, t in enumerate(texts):
            vec = state.hybrid_rag.embedder.encode_text(t)
            data.append({"object": "embedding", "index": idx, "embedding": vec})
        return {
            "object": "list",
            "model": "AnkaLM-BiEncoder-Embed-v1 (Trained From Scratch)",
            "data": data,
        }

    @app.post("/tokenize")
    @app.post("/api/tokenize")
    def tokenize_endpoint(req: TokenizeRequest) -> Dict[str, Any]:
        tok = state.get_tokenizer()
        ids = tok.encode(req.text, add_bos=req.add_bos, add_eos=req.add_eos)
        tokens = [tok.id_to_token(i) for i in ids]
        return {
            "text": req.text,
            "token_ids": ids,
            "tokens": tokens,
            "token_count": len(ids),
            "vocab_size": tok.vocab_size,
        }

    @app.post("/api/detokenize")
    def detokenize_endpoint(req: DetokenizeRequest) -> Dict[str, Any]:
        tok = state.get_tokenizer()
        text = tok.decode(req.token_ids, skip_special_tokens=req.skip_special_tokens)
        return {"token_ids": req.token_ids, "text": text}

    def _build_prompt_from_messages(messages: List[ChatMessageInput], rag_chunks: List[Dict[str, Any]]) -> str:
        parts = ["<|bos|>"]
        has_system = any(m.role == "system" for m in messages)
        if not has_system:
            parts.append("<|system|>\nSen AnkaLM, yerel donanımda sıfırdan eğitilmiş Türkçe yapay zeka asistanısın.<|eot|>")
        for msg in messages:
            parts.append(f"<|{msg.role}|>\n{msg.content}<|eot|>")
        if rag_chunks:
            rag_context = "\n".join(f"{c['citation']}: {c['content']}" for c in rag_chunks)
            parts.insert(1, f"<|tool|>\n[Yerel Belge Bilgisi - Öğrenilmiş Model Ağırlıklarından Ayrıdır]\n{rag_context}<|eot|>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    @app.post("/api/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
        try:
            engine = state.get_or_load_engine(req.model)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), req.messages[-1].content)
        rag_citations: List[Dict[str, Any]] = []
        if req.use_rag:
            rag_citations = state.hybrid_rag.search(last_user_msg, top_k=3, workspace=req.workspace)

        prompt = _build_prompt_from_messages(req.messages, rag_citations)
        gen_cfg = GenerationConfig(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            greedy=req.greedy,
            repetition_penalty=req.repetition_penalty,
            seed=req.seed,
        )

        if req.conversation_id:
            try:
                state.chat_db.add_message(req.conversation_id, "user", last_user_msg)
            except Exception:
                pass

        comp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if not req.stream:
            spec_meta = None
            if req.speculative_decoding and (root / "checkpoints" / "edge_final" / "metadata.json").exists():
                draft_eng = AnkaLMInferenceEngine.from_checkpoint("edge_final", checkpoints_dir=root / "checkpoints", tokenizer_dir=root / "tokenizer")
                spec_dec = SpeculativeDecoder(draft_eng, engine, lookahead_k=3)
                s_res = spec_dec.generate_speculative(prompt, max_new_tokens=req.max_tokens)
                content = s_res.generated_text.strip()
                tps = s_res.speculative_tokens_per_sec
                comp_tokens = s_res.total_generated_tokens
                prompt_tokens = len(engine.tokenizer.encode(prompt))
                first_tok_ms = 5.0
                finish_reason = "stop"
                spec_meta = s_res.to_dict()
            else:
                res = engine.generate(prompt, gen_cfg=gen_cfg)
                content = res.generated_text.strip()
                tps = res.tokens_per_second
                comp_tokens = res.completion_tokens
                prompt_tokens = res.prompt_tokens
                first_tok_ms = res.first_token_latency_ms
                finish_reason = res.finish_reason

            if rag_citations:
                cite_header = "\n\n---\n**Yerel Hibrit RAG Belge Kaynakları (Model Eğitiminden Bağımsız):**\n" + "\n".join(
                    f"- {c['citation']} (Skor: {c.get('hybrid_score', 0):.2f})" for c in rag_citations
                )
                content = f"{content}{cite_header}"

            if req.conversation_id:
                try:
                    state.chat_db.add_message(
                        req.conversation_id,
                        "assistant",
                        content,
                        rag_citations=rag_citations,
                        metrics={
                            "tokens_per_second": tps,
                            "first_token_latency_ms": first_tok_ms,
                            "completion_tokens": comp_tokens,
                            "maturity_level": engine.metadata.maturity_level,
                        },
                    )
                except Exception:
                    pass

            return {
                "id": comp_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": engine.metadata.checkpoint_name,
                "model_name": engine.metadata.model_name,
                "maturity_level": engine.metadata.maturity_level,
                "maturity_description": engine.metadata.maturity_description,
                "rag_citations": rag_citations,
                "speculative_decoding": spec_meta,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "total_tokens": prompt_tokens + comp_tokens,
                    "tokens_per_second": tps,
                    "first_token_latency_ms": first_tok_ms,
                },
            }

        def _sse_generator() -> Iterator[str]:
            full_text = ""
            final_meta: Dict[str, Any] = {}
            for chunk in engine.generate_stream(prompt, gen_cfg=gen_cfg):
                delta = chunk.get("delta", "")
                full_text = chunk.get("text", full_text)
                if chunk.get("finished"):
                    final_meta = chunk
                payload = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": engine.metadata.checkpoint_name,
                    "maturity_level": engine.metadata.maturity_level,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta} if delta else {},
                            "finish_reason": chunk.get("finish_reason") if chunk.get("finished") else None,
                        }
                    ],
                }
                if chunk.get("finished"):
                    payload["usage"] = {
                        "prompt_tokens": chunk.get("prompt_tokens", 0),
                        "completion_tokens": chunk.get("completion_tokens", 0),
                        "total_tokens": chunk.get("total_tokens", 0),
                        "tokens_per_second": chunk.get("tokens_per_second", 0.0),
                        "first_token_latency_ms": chunk.get("first_token_latency_ms", 0.0),
                    }
                    payload["rag_citations"] = rag_citations
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if req.conversation_id:
                try:
                    state.chat_db.add_message(
                        req.conversation_id,
                        "assistant",
                        full_text.strip(),
                        rag_citations=rag_citations,
                        metrics={
                            "tokens_per_second": final_meta.get("tokens_per_second", 0.0),
                            "first_token_latency_ms": final_meta.get("first_token_latency_ms", 0.0),
                            "completion_tokens": final_meta.get("completion_tokens", 0),
                            "maturity_level": engine.metadata.maturity_level,
                        },
                    )
                except Exception:
                    pass
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    # --- MULTIMODAL DOCUMENT, PDF, EXCEL, IMAGE, AUDIO & HYBRID RAG ENDPOINTS (Section 18) ---
    @app.post("/documents/upload")
    @app.post("/api/rag/upload")
    async def upload_document_endpoint(
        file: UploadFile = File(...),
        workspace: str = Form("general"),
    ) -> Dict[str, Any]:
        raw = await file.read()
        fname = file.filename or "belge.txt"
        try:
            res = state.doc_pipeline.ingest_file(raw, filename=fname, workspace=workspace)
            d = res.to_dict()
            d["id"] = res.document_id
            d["chunk_count"] = res.page_or_sheet_or_slide_count
            return d
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/documents/list")
    @app.get("/api/rag/documents")
    def list_documents_endpoint(workspace: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
        return state.hybrid_rag.list_documents(workspace=workspace)

    @app.delete("/api/rag/documents/{doc_id}")
    def delete_document_endpoint(doc_id: str) -> Dict[str, Any]:
        return {"deleted": state.hybrid_rag.delete_document(doc_id), "id": doc_id}

    @app.post("/documents/search")
    @app.post("/api/rag/query")
    def search_documents_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        q = str(payload.get("query", ""))
        top_k = int(payload.get("top_k", 4))
        ws = payload.get("workspace")
        tag = payload.get("tag")
        return {"query": q, "results": state.hybrid_rag.search(q, top_k=top_k, workspace=ws, tag_filter=tag)}

    @app.post("/documents/analyze")
    def analyze_document_question(req: DocumentQueryRequest) -> Dict[str, Any]:
        return state.doc_pipeline.query_document_or_collection(
            question=req.question,
            filename=req.filename,
            workspace=req.workspace,
            top_k=req.top_k,
        )

    @app.post("/spreadsheets/analyze")
    async def analyze_spreadsheet_endpoint(
        file: UploadFile = File(...),
        question: str = Form("Bu Excel dosyasını özetle."),
    ) -> Dict[str, Any]:
        raw = await file.read()
        fname = file.filename or "tablo.xlsx"
        ing = state.doc_pipeline.ingest_file(raw, filename=fname, workspace="spreadsheet")
        qa = state.doc_pipeline.query_document_or_collection(question, filename=fname, workspace="spreadsheet")
        return {"ingestion": ing.to_dict(), "qa": qa}

    @app.post("/images/analyze")
    async def analyze_image_endpoint(
        file: UploadFile = File(...),
        question: str = Form("Bu resimde ne var?"),
    ) -> Dict[str, Any]:
        raw = await file.read()
        fname = file.filename or "gorsel.png"
        ing = state.doc_pipeline.ingest_file(raw, filename=fname, workspace="image")
        qa = state.doc_pipeline.query_document_or_collection(question, filename=fname, workspace="image")
        return {"ingestion": ing.to_dict(), "qa": qa}

    @app.post("/audio/analyze")
    async def analyze_audio_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
        raw = await file.read()
        fname = file.filename or "ses.wav"
        ing = state.doc_pipeline.ingest_file(raw, filename=fname, workspace="audio")
        return ing.to_dict()

    # --- CONTROLLED TOOL CALLING & PYTHON SANDBOX ENDPOINTS (Section 10) ---
    @app.get("/tools/list")
    @app.get("/api/tools")
    def list_tools_endpoint() -> List[Dict[str, Any]]:
        return state.tool_registry.list_tools()

    @app.post("/tools/execute")
    @app.post("/api/tools/execute")
    def execute_tool_endpoint(req: ToolExecuteRequest) -> Dict[str, Any]:
        resp = state.tool_registry.execute(
            ToolCallRequest(
                tool_name=req.tool_name,
                arguments=req.arguments,
                user_authorized=req.user_authorized,
                timeout_sec=req.timeout_sec,
            )
        )
        return resp.to_dict()

    # --- Chat History Endpoints ---
    @app.get("/api/conversations")
    def list_conversations() -> List[Dict[str, Any]]:
        return state.chat_db.list_conversations()

    @app.post("/api/conversations")
    def create_conversation(req: ConversationCreateRequest) -> Dict[str, Any]:
        return state.chat_db.create_conversation(
            title=req.title,
            system_prompt=req.system_prompt,
            model_checkpoint=req.model_checkpoint,
        )

    @app.get("/api/conversations/search")
    def search_conversations(q: str = Query("")) -> List[Dict[str, Any]]:
        return state.chat_db.search_conversations(q)

    @app.get("/api/conversations/{cid}")
    def get_conversation(cid: str) -> Dict[str, Any]:
        try:
            return state.chat_db.get_conversation(cid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/api/conversations/{cid}")
    def update_conversation(cid: str, req: ConversationUpdateRequest) -> Dict[str, Any]:
        try:
            return state.chat_db.rename_conversation(
                cid,
                title=req.title,
                system_prompt=req.system_prompt,
                model_checkpoint=req.model_checkpoint,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/conversations/{cid}")
    def delete_conversation(cid: str) -> Dict[str, Any]:
        return {"deleted": state.chat_db.delete_conversation(cid), "id": cid}

    @app.get("/api/conversations/{cid}/export")
    def export_conversation(cid: str) -> Dict[str, Any]:
        try:
            return state.chat_db.export_conversation(cid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/conversations/import")
    def import_conversation(payload: Dict[str, Any]) -> Dict[str, Any]:
        return state.chat_db.import_conversation(payload)

    @app.put("/api/messages/{mid}")
    def edit_message(mid: str, req: MessageEditRequest) -> Dict[str, Any]:
        try:
            return state.chat_db.edit_message(mid, req.content, truncate_after=req.truncate_after)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/messages/{mid}")
    def delete_message(mid: str) -> Dict[str, Any]:
        return {"deleted": state.chat_db.delete_message(mid), "id": mid}

    # --- Training Control Center & Research Reports Endpoints ---
    @app.get("/training/status")
    @app.get("/api/training/state")
    def get_training_state() -> Dict[str, Any]:
        live = state.proc_manager.read_state()
        history: List[Dict[str, Any]] = []
        m_file = state.proc_manager.metrics_file
        if m_file.exists():
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-120:]
                history = [json.loads(line) for line in lines if line.strip()]
            except Exception:
                pass
        return {"live": live, "metrics_history": history}

    @app.get("/api/training/logs")
    def get_training_logs(lines: int = 80) -> Dict[str, Any]:
        log_file = root / "logs" / "ankalm.log"
        content: List[str] = []
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                content = [l.rstrip("\n") for l in f.readlines()[-lines:]]
        return {"log_file": str(log_file), "lines": content}

    @app.get("/api/training/configs")
    def list_training_configs() -> List[Dict[str, Any]]:
        cfg_dir = root / "configs"
        configs = []
        if cfg_dir.exists():
            for p in sorted(cfg_dir.glob("*.yaml")):
                try:
                    c = AnkaLMConfig.from_yaml(p)
                    tot_p, act_p = c.model.estimate_total_and_active_parameters()
                    configs.append({
                        "filename": p.name,
                        "path": f"configs/{p.name}",
                        "model_name": c.model.name,
                        "profile": c.model.profile,
                        "estimated_params": tot_p,
                        "active_params": act_p,
                        "use_moe": c.model.use_moe,
                        "max_steps": c.training.max_steps,
                        "batch_size": c.training.batch_size,
                        "precision": c.training.precision,
                    })
                except Exception:
                    pass
        return configs

    @app.post("/api/training/start")
    def start_training(req: TrainingStartRequest) -> Dict[str, Any]:
        safe_cfg = validate_safe_path(req.config_path, root)
        if not safe_cfg.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {req.config_path}")
        try:
            pid = state.proc_manager.start_background_training(
                config_path=req.config_path,
                resume_from=req.resume_from,
                max_steps=req.max_steps,
            )
            return {"status": "started", "pid": pid, "config_path": req.config_path}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/training/stop")
    def stop_training() -> Dict[str, Any]:
        state.proc_manager.request_stop()
        return {"status": "stop_requested"}

    @app.post("/api/training/pause")
    def pause_or_resume_training(payload: Dict[str, Any]) -> Dict[str, Any]:
        paused = bool(payload.get("paused", not state.proc_manager.is_paused()))
        state.proc_manager.request_pause(paused=paused)
        return {"status": "paused" if paused else "resumed", "paused": paused}

    @app.post("/api/training/resume")
    def resume_training_endpoint(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = payload or {}
        cfg_path = str(data.get("config_path", "configs/small.yaml"))
        ckpt = str(data.get("checkpoint", "latest"))
        try:
            pid = state.proc_manager.start_background_training(config_path=cfg_path, resume_from=ckpt)
            return {"status": "resumed_in_background", "pid": pid, "checkpoint": ckpt}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/training/save-checkpoint")
    def request_save_checkpoint() -> Dict[str, Any]:
        state.proc_manager.request_immediate_checkpoint()
        return {"status": "checkpoint_save_requested"}

    @app.post("/api/training/evaluate")
    def trigger_evaluation(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = payload or {}
        ckpt = str(data.get("checkpoint", "latest"))
        try:
            rep = run_turkish_evaluation_lab(
                checkpoint_name=ckpt,
                checkpoints_dir=root / "checkpoints",
                tokenizer_dir=root / "tokenizer",
            )
            return rep.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/evaluation/report")
    def get_evaluation_report() -> Dict[str, Any]:
        rep_path = root / "logs" / "evaluation_report.json"
        if rep_path.exists():
            with open(rep_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "not_run_yet"}

    @app.get("/api/moe/report")
    def get_moe_report() -> Dict[str, Any]:
        p = root / "logs" / "moe_training_and_routing_report.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"status": "not_available"}

    @app.get("/api/low-vram/report")
    def get_low_vram_report() -> Dict[str, Any]:
        p = root / "logs" / "low_vram_and_speculative_report.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"status": "not_available"}

    @app.get("/api/family/report")
    def get_family_report() -> List[Dict[str, Any]]:
        p = root / "logs" / "model_family_report.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return []

    @app.get("/api/hardware/report")
    def get_hardware_report() -> Dict[str, Any]:
        hw_path = root / "logs" / "hardware_report.json"
        if hw_path.exists():
            with open(hw_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return discover_hardware().to_dict()

    @app.get("/api/storage/summary")
    def get_storage_summary() -> Dict[str, Any]:
        return state.storage_manager.inspect_storage().to_dict()

    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        idx = web_dir / "index.html"
        if idx.exists():
            return FileResponse(idx)
        return HTMLResponse("<h1>AnkaLM AI Studio Local API is running on 127.0.0.1</h1>")

    return app


def run_local_server(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Launch the AnkaLM localhost-only API and AI Studio server."""
    verify_localhost_binding(host)
    import uvicorn

    app = create_app(PROJECT_ROOT)
    uvicorn.run(app, host=host, port=port, log_level="info")
