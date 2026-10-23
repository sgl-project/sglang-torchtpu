//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod pd_routing_tests {
    use super::*;

    /// Test basic PD mode routing with prefill and decode workers
    #[tokio::test]
    async fn test_pd_mode_basic_routing() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://127.0.0.1:19800".to_string(), None),
                    ("http://127.0.0.1:19801".to_string(), None),
                ],
                vec![
                    "http://127.0.0.1:19802".to_string(),
                    "http://127.0.0.1:19803".to_string(),
                ],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3800)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        // Note: For PD mode tests, we need to start prefill and decode workers separately
        // The test context will need to handle this specially
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill workers
                TestWorkerConfig::prefill(19800),
                TestWorkerConfig::prefill(19801),
                // Decode workers
                TestWorkerConfig::decode(19802),
                TestWorkerConfig::decode(19803),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        for i in 0..10 {
            let payload = json!({
                "text": format!("PD mode request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "PD mode request should succeed"
            );
        }

        ctx.shutdown().await;
    }

    /// Test PD mode with round robin policy
    #[tokio::test]
    async fn test_pd_mode_round_robin() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19810".to_string(), None)],
                vec![
                    "http://127.0.0.1:19811".to_string(),
                    "http://127.0.0.1:19812".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3801)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19810),
                TestWorkerConfig::decode(19811),
                TestWorkerConfig::decode(19812),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        for i in 0..20 {
            let payload = json!({
                "text": format!("PD round robin {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed in PD mode with round robin"
        );

        ctx.shutdown().await;
    }

    /// Test PD mode handles worker failures gracefully
    #[tokio::test]
    async fn test_pd_mode_with_failing_decode_worker() {
        use smg::config::RetryConfig;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19820".to_string(), None)],
                vec![
                    "http://127.0.0.1:19821".to_string(),
                    "http://127.0.0.1:19822".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3802)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19820),
                MockWorkerConfig {
                    port: 19821,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0, // Failing decode worker
                },
                TestWorkerConfig::decode(19822), // Healthy decode worker
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Request should succeed via retry to healthy decode worker
        let payload = json!({
            "text": "Test with failing decode worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry to healthy decode worker"
        );

        ctx.shutdown().await;
    }

    /// Prefill takes this long to answer; a router that waits for it after an
    /// early decode failure would take at least this long to respond.
    const SLOW_PREFILL_DELAY_MS: u64 = 3000;
    /// Generous bound for an early-returning router, well below the prefill delay.
    const EARLY_RETURN_BOUND_MS: u128 = 1500;

    /// Sends one request to a PD router whose only prefill is slow and whose only
    /// decode always fails immediately, and returns (status, elapsed_ms).
    async fn run_early_decode_failure(
        prefill_port: u16,
        decode_port: u16,
        router_port: u16,
        max_retries: u32,
        stream: bool,
    ) -> (StatusCode, u128) {
        use smg::config::RetryConfig;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{prefill_port}"), None)],
                vec![format!("http://127.0.0.1:{decode_port}")],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(router_port)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                MockWorkerConfig {
                    port: prefill_port,
                    worker_type: WorkerType::Prefill,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: SLOW_PREFILL_DELAY_MS,
                    fail_rate: 0.0,
                },
                MockWorkerConfig {
                    port: decode_port,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0,
                },
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let payload = json!({
            "text": "Early decode failure must not wait for prefill",
            "stream": stream
        });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let start = std::time::Instant::now();
        let resp = app.oneshot(req).await.unwrap();
        let elapsed_ms = start.elapsed().as_millis();
        let status = resp.status();

        ctx.shutdown().await;
        (status, elapsed_ms)
    }

    /// #658: when decode rejects a request before prefill finishes, the router
    /// must return immediately (dropping the prefill connection) instead of
    /// holding prefill compute until prefill completes.
    #[tokio::test]
    async fn test_early_decode_failure_does_not_wait_for_prefill() {
        let (status, elapsed_ms) = run_early_decode_failure(19860, 19861, 3806, 0, false).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            elapsed_ms < EARLY_RETURN_BOUND_MS,
            "router waited {elapsed_ms}ms for a doomed prefill (prefill delay {SLOW_PREFILL_DELAY_MS}ms)"
        );
    }

    /// Streaming variant of the #658 early-return behavior.
    #[tokio::test]
    async fn test_early_decode_failure_does_not_wait_for_prefill_streaming() {
        let (status, elapsed_ms) = run_early_decode_failure(19870, 19871, 3807, 0, true).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            elapsed_ms < EARLY_RETURN_BOUND_MS,
            "router waited {elapsed_ms}ms for a doomed prefill (prefill delay {SLOW_PREFILL_DELAY_MS}ms)"
        );
    }

    /// With retries enabled (the production default), every attempt must return
    /// early, so the total stays far below even a single prefill delay.
    #[tokio::test]
    async fn test_early_decode_failure_with_retries_does_not_wait_for_prefill() {
        let (status, elapsed_ms) = run_early_decode_failure(19880, 19881, 3808, 2, false).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            elapsed_ms < EARLY_RETURN_BOUND_MS,
            "3 attempts took {elapsed_ms}ms; at least one waited for prefill (delay {SLOW_PREFILL_DELAY_MS}ms)"
        );
    }
}

#[cfg(test)]
mod pd_responses_routing_tests {
    use std::{sync::Arc, time::Duration};

    use axum::routing::post;
    use http_body_util::BodyExt;
    use smg::{
        config::{PolicyConfig, RetryConfig},
        core::{
            BasicWorkerBuilder, DPAwareWorkerBuilder, Worker, WorkerRegistry,
            WorkerType as CoreWorkerType,
        },
        policies::PolicyRegistry,
        protocols::responses::ResponsesRequest,
        routers::{pd_router::PDRouter, RouterTrait},
    };
    use tokio::{
        io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
        net::{TcpListener, TcpStream},
        sync::{oneshot, Mutex},
        task::JoinHandle,
        time::timeout,
    };

    use super::*;

    const STORE_ERROR: &str = r#"{"error":{"message":"Response store is disabled. Stateful Responses require --enable-response-store on a standalone server; response storage is unavailable in PD mode.","type":"invalid_request_error","param":"previous_response_id","code":400}}"#;
    const DECODE_RESPONSE: &str =
        r#"{"id":"resp_decode","object":"response","status":"completed"}"#;
    const DECODE_STREAM: &str = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_decode\",\"status\":\"completed\"}}\n\n";

    #[derive(Clone, Copy)]
    enum WorkerReply {
        Json(StatusCode, &'static str),
        Stream(&'static str),
    }

    /// Regression test: `/v1/responses` must be routed through the PD
    /// dual-dispatch path instead of falling back to the `RouterTrait`
    /// default `501 NOT_IMPLEMENTED` implementation.
    #[tokio::test]
    async fn test_pd_mode_responses_routing() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19830".to_string(), None)],
                vec!["http://127.0.0.1:19831".to_string()],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3803)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19830),
                TestWorkerConfig::decode(19831),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "input": "PD mode responses request",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "PD mode /v1/responses request should be dual-dispatched, not 501"
        );

        ctx.shutdown().await;
    }

    fn make_pd_router() -> PDRouter {
        PDRouter {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            client: reqwest::Client::new(),
            retry_config: RetryConfig::default(),
            api_key: None,
            enable_igw: false,
        }
    }

    fn register_workers(router: &PDRouter, prefill_url: String, decode_url: String) {
        let prefill = DPAwareWorkerBuilder::new(prefill_url, 2, 4)
            .worker_type(CoreWorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        prefill.set_healthy(true);
        let decode = BasicWorkerBuilder::new(decode_url)
            .worker_type(CoreWorkerType::Decode)
            .build();
        decode.set_healthy(true);
        router.worker_registry.register(Arc::new(prefill));
        router.worker_registry.register(Arc::new(decode));
    }

    /// Spawn a local worker that records every JSON body POSTed to
    /// `/v1/responses` and returns either a JSON or an SSE response.
    async fn spawn_capture_worker(
        captured: Arc<Mutex<Vec<serde_json::Value>>>,
        reply: WorkerReply,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/v1/responses",
            post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let captured = captured.clone();
                async move {
                    // SRT forwards Responses.stream to ChatCompletionRequest's non-nullable bool.
                    let stream = body.get("stream").cloned().unwrap_or(json!(false));
                    if serde_json::from_value::<bool>(stream).is_err() {
                        return axum::response::Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .header(CONTENT_TYPE, "application/json")
                            .body(Body::from(
                                r#"{"error":{"message":"stream must be a boolean"}}"#,
                            ))
                            .unwrap();
                    }
                    captured.lock().await.push(body);
                    let (status, content_type, body) = match reply {
                        WorkerReply::Json(status, body) => (status, "application/json", body),
                        WorkerReply::Stream(body) => (StatusCode::OK, "text/event-stream", body),
                    };
                    axum::response::Response::builder()
                        .status(status)
                        .header(CONTENT_TYPE, content_type)
                        .header("content-length", body.len())
                        .body(Body::from(body))
                        .unwrap()
                }
            }),
        );
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), task)
    }

    fn responses_request(payload: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(payload).expect("valid responses request")
    }

    /// PDRouter must inject the PD bootstrap metadata into both worker
    /// requests, forward `disagg_prefill_dp_rank` to decode for a DP-aware
    /// prefill worker, target `/v1/responses` on both workers, and return the
    /// decode response.
    #[tokio::test]
    async fn test_pd_responses_default_buffered_request_injects_bootstrap_metadata() {
        let prefill_bodies = Arc::new(Mutex::new(Vec::new()));
        let decode_bodies = Arc::new(Mutex::new(Vec::new()));
        let (prefill_url, prefill_task) = spawn_capture_worker(
            prefill_bodies.clone(),
            WorkerReply::Json(StatusCode::OK, r#"{"id":"resp_prefill"}"#),
        )
        .await;
        let (decode_url, decode_task) = spawn_capture_worker(
            decode_bodies.clone(),
            WorkerReply::Json(StatusCode::OK, DECODE_RESPONSE),
        )
        .await;
        let router = make_pd_router();
        register_workers(&router, prefill_url, decode_url);
        let request = responses_request(json!({
            "model": "mock-model", "input": "Hello PD responses"
        }));
        assert_eq!(request.stream, None);

        let body = timeout(Duration::from_secs(5), async {
            let response = router.route_responses(None, &request, None).await;
            assert_eq!(response.status(), StatusCode::OK);
            response.into_body().collect().await.unwrap().to_bytes()
        })
        .await
        .expect("PD response must complete");
        assert_eq!(body.as_ref(), DECODE_RESPONSE.as_bytes());

        let prefill_bodies = prefill_bodies.lock().await;
        let decode_bodies = decode_bodies.lock().await;
        assert_eq!(prefill_bodies.len(), 1);
        assert_eq!(decode_bodies.len(), 1);
        let prefill_body = &prefill_bodies[0];
        let decode_body = &decode_bodies[0];
        for body in [prefill_body, decode_body] {
            assert_eq!(body["stream"], false);
            assert_eq!(body["input"], "Hello PD responses");
            assert_eq!(body["model"], "mock-model");
            assert_eq!(body["bootstrap_host"], "127.0.0.1");
            assert_eq!(body["bootstrap_port"], 8998);
            assert!(body["bootstrap_room"].is_u64());
        }
        assert_eq!(
            decode_body["bootstrap_room"],
            prefill_body["bootstrap_room"]
        );
        assert_eq!(prefill_body["data_parallel_rank"], 2);
        assert!(prefill_body.get("disagg_prefill_dp_rank").is_none());
        assert_eq!(decode_body["disagg_prefill_dp_rank"], 2);
        prefill_task.abort();
        decode_task.abort();
    }

    /// Streaming Responses requests must flow through the PD dual-dispatch
    /// path and return the decode SSE stream unchanged, with bootstrap
    /// metadata still injected into both worker requests.
    #[tokio::test]
    async fn test_pd_responses_streaming_passthrough() {
        let prefill_bodies = Arc::new(Mutex::new(Vec::new()));
        let decode_bodies = Arc::new(Mutex::new(Vec::new()));
        let (prefill_url, prefill_task) = spawn_capture_worker(
            prefill_bodies.clone(),
            WorkerReply::Stream("data: {\"id\":\"resp_prefill\"}\n\n"),
        )
        .await;
        let (decode_url, decode_task) =
            spawn_capture_worker(decode_bodies.clone(), WorkerReply::Stream(DECODE_STREAM)).await;
        let router = make_pd_router();
        register_workers(&router, prefill_url, decode_url);
        let request = responses_request(json!({
            "model": "mock-model", "input": "Hello PD streaming", "stream": true
        }));

        let body = timeout(Duration::from_secs(5), async {
            let response = router.route_responses(None, &request, None).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
            response.into_body().collect().await.unwrap().to_bytes()
        })
        .await
        .expect("PD stream must complete without a [DONE] sentinel");
        assert_eq!(body.as_ref(), DECODE_STREAM.as_bytes());
        let prefill_bodies = prefill_bodies.lock().await;
        let decode_bodies = decode_bodies.lock().await;
        assert_eq!(prefill_bodies.len(), 1);
        assert_eq!(decode_bodies.len(), 1);
        assert_eq!(decode_bodies[0]["stream"], true);
        assert!(prefill_bodies[0]["bootstrap_room"].is_u64());
        assert_eq!(
            decode_bodies[0]["bootstrap_room"],
            prefill_bodies[0]["bootstrap_room"]
        );
        prefill_task.abort();
        decode_task.abort();
    }

    #[tokio::test]
    async fn test_pd_responses_decode_error_is_valid_sse() {
        let (prefill_url, prefill_task) = spawn_capture_worker(
            Arc::new(Mutex::new(Vec::new())),
            WorkerReply::Json(StatusCode::OK, "{}"),
        )
        .await;
        let (decode_url, decode_task) = spawn_capture_worker(
            Arc::new(Mutex::new(Vec::new())),
            WorkerReply::Json(StatusCode::BAD_REQUEST, STORE_ERROR),
        )
        .await;
        let router = make_pd_router();
        register_workers(&router, prefill_url, decode_url);
        let request = responses_request(json!({
            "model":"mock-model", "input":"hi", "stream":true
        }));
        let body = timeout(Duration::from_secs(5), async {
            let response = router.route_responses(None, &request, None).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
            assert!(!response.headers().contains_key("content-length"));
            response.into_body().collect().await.unwrap().to_bytes()
        })
        .await
        .expect("decode error must complete");
        let body = std::str::from_utf8(&body).unwrap();
        let data = body
            .strip_suffix("\n\n")
            .expect("SSE event delimiter")
            .strip_prefix("data: ")
            .unwrap();
        let data: serde_json::Value = serde_json::from_str(data).unwrap();
        assert_eq!(data["error"]["status"], 400);
        assert_eq!(
            data["error"]["message"],
            serde_json::from_str::<serde_json::Value>(STORE_ERROR).unwrap()
        );
        prefill_task.abort();
        decode_task.abort();
    }

    /// Read the gateway's fixed-length JSON POST without an HTTP server that
    /// could hide disconnects by keeping a pending handler alive.
    async fn read_responses_post(socket: &mut BufReader<TcpStream>) -> serde_json::Value {
        let mut line = String::new();
        socket.read_line(&mut line).await.unwrap();
        assert_eq!(line, "POST /v1/responses HTTP/1.1\r\n");
        let mut content_length = None;
        loop {
            line.clear();
            assert_ne!(socket.read_line(&mut line).await.unwrap(), 0);
            if line == "\r\n" {
                break;
            }
            let (name, value) = line.split_once(':').unwrap();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = Some(value.trim().parse::<usize>().unwrap());
            }
        }
        let mut body = vec![0; content_length.expect("JSON POST must have Content-Length")];
        socket.read_exact(&mut body).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// Model #39122's pre-generation admission error without depending on its
    /// serving implementation. Prefill waits until decode has received the
    /// request, then rejects it without producing KV. Decode sends no headers
    /// or KV-timeout response: only a router disconnect can finish its task.
    #[tokio::test]
    async fn test_pd_responses_stateful_400_cancels_pending_decode() {
        for (param, value, stream) in [
            ("previous_response_id", json!("resp_previous"), false),
            ("previous_response_id", json!("resp_previous"), true),
            ("background", json!(true), true),
        ] {
            let prefill_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let decode_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let prefill_url = format!("http://{}", prefill_listener.local_addr().unwrap());
            let decode_url = format!("http://{}", decode_listener.local_addr().unwrap());
            let (decode_started_tx, decode_started_rx) = oneshot::channel();
            let admission_error = json!({
                "error": {
                    "message": "Response store is disabled. Stateful Responses require --enable-response-store on a standalone server; response storage is unavailable in PD mode.",
                    "type": "BadRequestError",
                    "param": param,
                    "code": 400
                }
            });
            let error_body = admission_error.to_string();
            let mut prefill_task = tokio::spawn(async move {
                let (socket, _) = prefill_listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                let body = read_responses_post(&mut socket).await;
                decode_started_rx.await.unwrap();
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    error_body.len(), error_body
                );
                socket
                    .get_mut()
                    .write_all(response.as_bytes())
                    .await
                    .unwrap();
                body
            });
            let mut decode_task = tokio::spawn(async move {
                let (socket, _) = decode_listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                let body = read_responses_post(&mut socket).await;
                decode_started_tx.send(()).unwrap();
                let mut byte = [0];
                assert_eq!(
                    socket.read(&mut byte).await.unwrap(),
                    0,
                    "router must close the pending decode connection after prefill rejects"
                );
                body
            });

            let router = make_pd_router();
            let prefill = Arc::new(
                BasicWorkerBuilder::new(prefill_url)
                    .worker_type(CoreWorkerType::Prefill {
                        bootstrap_port: Some(9001),
                    })
                    .build(),
            );
            let decode = Arc::new(
                BasicWorkerBuilder::new(decode_url)
                    .worker_type(CoreWorkerType::Decode)
                    .build(),
            );
            prefill.set_healthy(true);
            decode.set_healthy(true);
            router.worker_registry.register(prefill.clone());
            router.worker_registry.register(decode.clone());
            let mut payload = json!({
                "model": "mock-model",
                "input": "Continue the conversation",
                "stream": stream
            });
            payload[param] = value.clone();
            let request = responses_request(payload);

            // The timeout is only a regression guard. No worker timer, client
            // timeout, test teardown, or router drop can release decode here.
            let result = timeout(Duration::from_secs(5), async {
                let response = router.route_responses(None, &request, None).await;
                let status = response.status();
                let error_code = response.headers().get("x-smg-error-code").cloned();
                let body = response.into_body().collect().await.unwrap().to_bytes();
                let (prefill_body, decode_body) = tokio::join!(&mut prefill_task, &mut decode_task);
                (
                    status,
                    error_code,
                    body,
                    prefill_body.unwrap(),
                    decode_body.unwrap(),
                )
            })
            .await;
            if result.is_err() {
                // join! may already have consumed a completed handle. Only
                // abort and await tasks that are still pending on failure.
                for task in [&mut prefill_task, &mut decode_task] {
                    if !task.is_finished() {
                        task.abort();
                        let _ = task.await;
                    }
                }
            }
            let (status, error_code, body, prefill_body, decode_body) = result.expect(
                "prefill admission error must finish routing and disconnect decode promptly",
            );
            assert_eq!(status, StatusCode::BAD_REQUEST, "{param}, stream={stream}");
            assert_eq!(error_code.unwrap(), "prefill_bad_request");
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert!(body["error"]["message"]
                .as_str()
                .unwrap()
                .contains(&admission_error.to_string()));
            assert_eq!(prefill_body[param], value);
            assert_eq!(decode_body[param], value);
            assert!(prefill_body["bootstrap_room"].is_u64());
            assert_eq!(
                prefill_body["bootstrap_room"],
                decode_body["bootstrap_room"]
            );
            assert_eq!(prefill.load(), 0);
            assert_eq!(decode.load(), 0);
        }
    }

    /// Detached background responses (`background=true, stream=false`) are
    /// retrieved through the /v1/responses/{id} endpoints, which the PD router
    /// does not implement. They must be rejected deterministically instead of
    /// dual-dispatched.
    #[tokio::test]
    async fn test_pd_responses_detached_background_rejected() {
        let router = make_pd_router();

        let request = responses_request(json!({
            "model": "mock-model",
            "input": "Hello background",
            "background": true,
            "stream": false
        }));

        let response = router.route_responses(None, &request, None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response
                .headers()
                .get("x-smg-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("pd_unsupported_background_responses")
        );
    }
}

/// #658: a streaming decode commits 200 headers before validating the request,
/// so the engine reports a rejection as an in-band SSE error event, not a
/// non-2xx. These workers reproduce that shape over real HTTP.
#[cfg(test)]
mod pd_early_stream_error_tests {
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::{Duration, Instant},
    };

    use axum::routing::post;
    use futures::StreamExt;
    use http_body_util::BodyExt;
    use smg::{
        config::{PolicyConfig, RetryConfig},
        core::{BasicWorkerBuilder, Worker, WorkerRegistry, WorkerType as CoreWorkerType},
        policies::PolicyRegistry,
        protocols::generate::GenerateRequest,
        routers::{pd_router::PDRouter, RouterTrait},
    };
    use tokio::{net::TcpListener, task::JoinHandle, time::timeout};

    use super::*;

    /// What an SGLang decode engine streams for an over-long prompt (observed
    /// on TPU v7x with a decode `--context-length` below the prompt length).
    const ENGINE_ERROR_STREAM: &str = "data: {\"error\":{\"message\":\"The input (2501 tokens) is longer than the model's context length (1024 tokens).\",\"type\":\"invalid_request_error\",\"code\":400,\"retryable\":false}}\n\ndata: [DONE]\n\n";
    const TOKEN_STREAM: &str = "data: {\"text\":\"tok\",\"meta_info\":{}}\n\ndata: [DONE]\n\n";
    const SLOW_PREFILL_MS: u64 = 3000;
    const EARLY_RETURN_BOUND: Duration = Duration::from_millis(1500);

    /// Prefill that completes after `delay_ms`, counting completions. With
    /// `headers_first` it behaves like a real engine serving a streaming
    /// request: 200 headers at once, body held open until prefill is done.
    async fn spawn_prefill(
        delay_ms: u64,
        headers_first: bool,
        completed: Arc<AtomicUsize>,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/generate",
            post(move || {
                let completed = completed.clone();
                async move {
                    let finish = async move {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, std::io::Error>(axum::body::Bytes::from("data: {}\n\n"))
                    };
                    let body = if headers_first {
                        Body::from_stream(futures::stream::once(finish))
                    } else {
                        Body::from(finish.await.unwrap())
                    };
                    axum::response::Response::builder()
                        .status(StatusCode::OK)
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(body)
                        .unwrap()
                }
            }),
        );
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), task)
    }

    /// Decode that immediately answers 200 SSE and streams `chunks`, pausing
    /// `gap_ms` between them.
    async fn spawn_decode(chunks: Vec<&'static str>, gap_ms: u64) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/generate",
            post(move || {
                let chunks = chunks.clone();
                async move {
                    let body = futures::stream::iter(chunks).then(move |chunk| async move {
                        tokio::time::sleep(Duration::from_millis(gap_ms)).await;
                        Ok::<_, std::io::Error>(axum::body::Bytes::from(chunk))
                    });
                    axum::response::Response::builder()
                        .status(StatusCode::OK)
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(body))
                        .unwrap()
                }
            }),
        );
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), task)
    }

    fn make_router(prefill_url: String, decode_url: String, max_retries: u32) -> PDRouter {
        let router = PDRouter {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            client: reqwest::Client::new(),
            retry_config: RetryConfig {
                max_retries,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
            api_key: None,
            enable_igw: false,
        };
        let prefill = BasicWorkerBuilder::new(prefill_url)
            .worker_type(CoreWorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        prefill.set_healthy(true);
        let decode = BasicWorkerBuilder::new(decode_url)
            .worker_type(CoreWorkerType::Decode)
            .build();
        decode.set_healthy(true);
        router.worker_registry.register(Arc::new(prefill));
        router.worker_registry.register(Arc::new(decode));
        router
    }

    /// Routes one streaming /generate and returns (status, body, elapsed).
    async fn stream_once(router: &PDRouter) -> (StatusCode, String, Duration) {
        let request: GenerateRequest =
            serde_json::from_value(json!({"text": "hello", "stream": true})).unwrap();
        let start = Instant::now();
        let (status, body) = timeout(Duration::from_secs(20), async {
            let response = router.route_generate(None, &request, None).await;
            let status = response.status();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        })
        .await
        .expect("PD stream must complete");
        (status, body, start.elapsed())
    }

    /// The in-band error is relayed unchanged without waiting for prefill.
    #[tokio::test]
    async fn test_in_band_decode_stream_error_does_not_wait_for_prefill() {
        let completed = Arc::new(AtomicUsize::new(0));
        let (prefill_url, p) = spawn_prefill(SLOW_PREFILL_MS, false, completed.clone()).await;
        let (decode_url, d) = spawn_decode(vec![ENGINE_ERROR_STREAM], 0).await;
        let router = make_router(prefill_url, decode_url, 0);

        let (status, body, elapsed) = stream_once(&router).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, ENGINE_ERROR_STREAM);
        assert!(
            elapsed < EARLY_RETURN_BOUND,
            "router waited {elapsed:?} for a doomed prefill (delay {SLOW_PREFILL_MS}ms)"
        );
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        p.abort();
        d.abort();
    }

    /// An error event split across chunks is still recognized, and all bytes
    /// read while peeking are relayed.
    #[tokio::test]
    async fn test_split_in_band_decode_stream_error_does_not_wait_for_prefill() {
        let completed = Arc::new(AtomicUsize::new(0));
        let (prefill_url, p) = spawn_prefill(SLOW_PREFILL_MS, false, completed).await;
        let (split_at, _) = ENGINE_ERROR_STREAM.split_once("\"message\"").unwrap();
        let rest = &ENGINE_ERROR_STREAM[split_at.len()..];
        let (decode_url, d) = spawn_decode(vec![split_at, rest], 50).await;
        let router = make_router(prefill_url, decode_url, 0);

        let (status, body, elapsed) = stream_once(&router).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, ENGINE_ERROR_STREAM);
        assert!(elapsed < EARLY_RETURN_BOUND, "router waited {elapsed:?}");
        p.abort();
        d.abort();
    }

    /// A 2xx decode stream that closes without an event is a decode fault; it
    /// must neither wait for prefill nor be relayed as a success.
    #[tokio::test]
    async fn test_empty_decode_stream_does_not_wait_for_prefill() {
        let completed = Arc::new(AtomicUsize::new(0));
        let (prefill_url, p) = spawn_prefill(SLOW_PREFILL_MS, false, completed).await;
        let (decode_url, d) = spawn_decode(vec![], 0).await;
        let router = make_router(prefill_url, decode_url, 0);

        let (status, _, elapsed) = stream_once(&router).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert!(elapsed < EARLY_RETURN_BOUND, "router waited {elapsed:?}");
        p.abort();
        d.abort();
    }

    /// Regression guard: a normal first event must not cancel prefill. The
    /// router still awaits prefill and relays the peeked bytes unchanged.
    #[tokio::test]
    async fn test_normal_decode_stream_still_awaits_prefill() {
        let prefill_delay_ms = 500;
        let completed = Arc::new(AtomicUsize::new(0));
        let (prefill_url, p) = spawn_prefill(prefill_delay_ms, false, completed.clone()).await;
        let (decode_url, d) = spawn_decode(vec![TOKEN_STREAM], 0).await;
        let router = make_router(prefill_url, decode_url, 0);

        let (status, body, elapsed) = stream_once(&router).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, TOKEN_STREAM);
        assert!(
            elapsed >= Duration::from_millis(prefill_delay_ms),
            "router returned after {elapsed:?} without awaiting prefill"
        );
        assert_eq!(completed.load(Ordering::SeqCst), 1);
        p.abort();
        d.abort();
    }

    /// A streaming prefill answers 200 headers at once; its completion is the
    /// end of its body. The in-band decode error must still win even when
    /// prefill's headers arrive first (the race seen on TPU v7x).
    #[tokio::test]
    async fn test_in_band_decode_error_with_streaming_prefill_does_not_wait() {
        let completed = Arc::new(AtomicUsize::new(0));
        let (prefill_url, p) = spawn_prefill(SLOW_PREFILL_MS, true, completed.clone()).await;
        // Decode's error trails prefill's headers.
        let (decode_url, d) = spawn_decode(vec![ENGINE_ERROR_STREAM], 100).await;
        let router = make_router(prefill_url, decode_url, 0);

        let (status, body, elapsed) = stream_once(&router).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, ENGINE_ERROR_STREAM);
        assert!(
            elapsed < EARLY_RETURN_BOUND,
            "router waited {elapsed:?} for a doomed prefill (delay {SLOW_PREFILL_MS}ms)"
        );
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        p.abort();
        d.abort();
    }

    /// Regression guard for the streaming-prefill shape: decode is relayed only
    /// after prefill's body ends, as before.
    #[tokio::test]
    async fn test_normal_decode_stream_awaits_streaming_prefill_body() {
        let prefill_delay_ms = 500;
        let completed = Arc::new(AtomicUsize::new(0));
        let (prefill_url, p) = spawn_prefill(prefill_delay_ms, true, completed.clone()).await;
        let (decode_url, d) = spawn_decode(vec![TOKEN_STREAM], 0).await;
        let router = make_router(prefill_url, decode_url, 0);

        let (status, body, elapsed) = stream_once(&router).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, TOKEN_STREAM);
        assert!(
            elapsed >= Duration::from_millis(prefill_delay_ms),
            "router returned after {elapsed:?} without awaiting prefill's body"
        );
        assert_eq!(completed.load(Ordering::SeqCst), 1);
        p.abort();
        d.abort();
    }
}

/// A retry reuses the client's `rid`. When an attempt ends with the router
/// cancelling the paired request (decode failed before prefill completed, or
/// prefill failed), the cancelled request stays registered on its engine until
/// the engine notices the disconnect (~4 s on TPU v7x). A retry that reaches
/// that engine in the meantime gets 400 "Duplicate request ID detected", which
/// is not retried, so the client would see it instead of the real error. The
/// router must return the real error instead, while requests without a `rid`
/// keep their retries.
#[cfg(test)]
mod pd_retry_rid_reuse_tests {
    use std::{
        collections::HashSet,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
        time::Duration,
    };

    use axum::{extract::State, routing::post, Json};
    use http_body_util::BodyExt;
    use serde_json::Value;
    use smg::{
        config::{PolicyConfig, RetryConfig},
        core::{BasicWorkerBuilder, Worker, WorkerRegistry, WorkerType as CoreWorkerType},
        policies::PolicyRegistry,
        protocols::generate::GenerateRequest,
        routers::{pd_router::PDRouter, RouterTrait},
    };
    use tokio::{net::TcpListener, task::JoinHandle, time::timeout};

    use super::*;

    /// Attempts the router makes per request, including the first.
    const MAX_ATTEMPTS: usize = 3;
    /// How long an engine keeps a rid registered after receiving it.
    const RID_HELD_MS: u64 = 3000;
    /// The failing engine answers after this delay, long before the other
    /// engine would.
    const REJECT_MS: u64 = 200;

    /// Engine that, like SRT's `TokenizerManager`, rejects a rid that is still
    /// registered with 400 "Duplicate request ID detected". It answers only
    /// after releasing the rid, so in these tests the router always cancels it.
    #[derive(Clone, Default)]
    struct RidHoldingEngine {
        in_flight: Arc<Mutex<HashSet<String>>>,
        duplicates: Arc<AtomicUsize>,
    }

    impl RidHoldingEngine {
        fn app(&self) -> axum::Router {
            axum::Router::new()
                .route("/generate", post(rid_holding_generate))
                .with_state(self.clone())
        }

        fn duplicates(&self) -> usize {
            self.duplicates.load(Ordering::SeqCst)
        }
    }

    async fn rid_holding_generate(
        State(engine): State<RidHoldingEngine>,
        Json(request): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        // Without a client rid the engine generates a unique one.
        if let Some(rid) = request["rid"].as_str().map(str::to_owned) {
            if !engine.in_flight.lock().unwrap().insert(rid.clone()) {
                engine.duplicates.fetch_add(1, Ordering::SeqCst);
                let message = format!("Duplicate request ID detected: {rid}");
                let error = json!({"error": {"message": message}});
                return (StatusCode::BAD_REQUEST, Json(error));
            }
            // The engine keeps the rid until the request finishes or, if the
            // router hangs up first, until the disconnect watcher aborts it.
            // Release it from a detached task, because axum drops this handler
            // when the router hangs up.
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(RID_HELD_MS)).await;
                engine.in_flight.lock().unwrap().remove(&rid);
            });
        }
        tokio::time::sleep(Duration::from_millis(RID_HELD_MS + 50)).await;
        (StatusCode::OK, Json(json!({"text": "", "meta_info": {}})))
    }

    /// Engine that rejects its first `failures` requests with a retryable 503,
    /// like an overloaded engine, and accepts the rest. A rejected request
    /// leaves no rid registered.
    #[derive(Clone)]
    struct FailingEngine {
        failures: usize,
        requests: Arc<AtomicUsize>,
    }

    impl FailingEngine {
        fn new(failures: usize) -> Self {
            Self {
                failures,
                requests: Arc::default(),
            }
        }

        fn app(&self) -> axum::Router {
            axum::Router::new()
                .route("/generate", post(failing_generate))
                .with_state(self.clone())
        }

        fn requests(&self) -> usize {
            self.requests.load(Ordering::SeqCst)
        }
    }

    async fn failing_generate(State(engine): State<FailingEngine>) -> (StatusCode, Json<Value>) {
        let seen = engine.requests.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(REJECT_MS)).await;
        if seen < engine.failures {
            let error = json!({"error": {"message": "engine overloaded"}});
            (StatusCode::SERVICE_UNAVAILABLE, Json(error))
        } else {
            (StatusCode::OK, Json(json!({"text": "", "meta_info": {}})))
        }
    }

    async fn spawn_server(app: axum::Router) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), task)
    }

    fn make_router(prefill_url: String, decode_url: String) -> PDRouter {
        let router = PDRouter {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            client: reqwest::Client::new(),
            retry_config: RetryConfig {
                max_retries: MAX_ATTEMPTS as u32,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
            api_key: None,
            enable_igw: false,
        };
        let prefill = BasicWorkerBuilder::new(prefill_url)
            .worker_type(CoreWorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        prefill.set_healthy(true);
        let decode = BasicWorkerBuilder::new(decode_url)
            .worker_type(CoreWorkerType::Decode)
            .build();
        decode.set_healthy(true);
        router.worker_registry.register(Arc::new(prefill));
        router.worker_registry.register(Arc::new(decode));
        router
    }

    /// Sends one non-streaming /generate through a router with one prefill and
    /// one decode, and returns the client's status and body. With a single
    /// worker per role every retry reaches the same engines; with several, the
    /// default cache_aware policy usually picks them again, because the retry's
    /// prompt matches what it recorded for the first attempt.
    async fn generate(
        prefill: axum::Router,
        decode: axum::Router,
        rid: Option<&str>,
    ) -> (StatusCode, String) {
        let (prefill_url, p) = spawn_server(prefill).await;
        let (decode_url, d) = spawn_server(decode).await;
        let router = make_router(prefill_url, decode_url);
        let mut request = json!({"text": "hello", "stream": false});
        if let Some(rid) = rid {
            request["rid"] = json!(rid);
        }
        let request: GenerateRequest = serde_json::from_value(request).unwrap();

        let result = timeout(Duration::from_secs(30), async {
            let response = router.route_generate(None, &request, None).await;
            let status = response.status();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8_lossy(&body).into_owned())
        })
        .await
        .expect("PD request must complete");
        p.abort();
        d.abort();
        result
    }

    /// A decode that fails before prefill completes cancels prefill, which
    /// keeps the rid registered.
    #[tokio::test]
    async fn test_early_decode_failure_with_client_rid_returns_decode_error() {
        let prefill = RidHoldingEngine::default();
        let decode = FailingEngine::new(usize::MAX);

        let (status, body) = generate(prefill.app(), decode.app(), Some("client-rid")).await;

        assert_eq!(
            prefill.duplicates(),
            0,
            "a retry reused the rid of the cancelled attempt still registered on prefill; \
             client got {status}: {body}"
        );
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "client must get decode's 503: {body}"
        );
        assert_eq!(decode.requests(), 1, "no retry while prefill holds the rid");
    }

    /// A failed prefill cancels decode, which keeps the rid registered. Without
    /// the guard, the retry's decode rejects the rid, and the client gets that
    /// 400 instead of prefill's error once prefill recovers.
    #[tokio::test]
    async fn test_prefill_failure_with_client_rid_returns_prefill_error() {
        let prefill = FailingEngine::new(1);
        let decode = RidHoldingEngine::default();

        let (status, body) = generate(prefill.app(), decode.app(), Some("client-rid")).await;

        assert_eq!(
            decode.duplicates(),
            0,
            "a retry reused the rid of the cancelled attempt still registered on decode; \
             client got {status}: {body}"
        );
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "client must get prefill's 503: {body}"
        );
        assert_eq!(prefill.requests(), 1, "no retry while decode holds the rid");
    }

    /// Without a client rid the engine gives every attempt a fresh id, so both
    /// cancel paths keep their retries.
    #[tokio::test]
    async fn test_paired_cancel_without_client_rid_still_retries() {
        let prefill = RidHoldingEngine::default();
        let decode = FailingEngine::new(usize::MAX);
        let (status, body) = generate(prefill.app(), decode.app(), None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert_eq!(decode.requests(), MAX_ATTEMPTS, "early decode failure");

        let prefill = FailingEngine::new(usize::MAX);
        let decode = RidHoldingEngine::default();
        let (status, body) = generate(prefill.app(), decode.app(), None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert_eq!(prefill.requests(), MAX_ATTEMPTS, "prefill failure");
    }
}
