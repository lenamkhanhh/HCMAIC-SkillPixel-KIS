# 07 — Known gaps and risks

| Gap/risk | Status/evidence | Next action |
|---|---|---|
| BTC data/rules/schema absent | BLOCKED | run 6-hour audit when released |
| FFmpeg/VFR runtime | INTERFACE_ONLY | execute real non-zero-start VFR fixture |
| PySceneDetect/TransNetV2 | INTERFACE_ONLY | implement only after paired shot study |
| SigLIP2/Jina CLIP v2 | INTERFACE_ONLY | approved cached-weight smoke + dimension check |
| Real OCR/ASR/caption | INTERFACE_ONLY | choose multilingual providers after data audit |
| Multimodal indexes connected to API | PARTIAL | build per-modality artifact/index adapter |
| Learned fusion/VLM rerank/KISC | DEFERRED | only after strong frozen baseline |
| CUDA and competition-scale latency | BLOCKED | benchmark on team GPU and real vector count |
| Official submission | BLOCKED | adapter + dry run after schema release |
| Browser E2E after UI feedback change | VERIFIED LOCALLY | repeat after any API/UI contract change |
| Duplicate charset-normalizer metadata | ENVIRONMENT DEFECT | clean rebuild venv, then rerun gates |
| Fixture/proxy overconfidence | PERMANENT RISK | retain evidence labels/disclaimers |

Technical debt: provider adapters are safe placeholders, the current benchmark
slice labels are unsliced, feedback is session-local/in-memory, and ANN
parameters are not yet persisted as a built artifact distinct from config.

The staged `--force` replacement has tested rollback, but it remains a
multi-file best-effort transaction: an operating-system failure during both
commit and rollback could still require manual recovery. The extended mapping
loader deliberately accepts legacy schemas, so strict rich-schema enforcement
must happen in the future BTC adapter rather than breaking current datasets.
