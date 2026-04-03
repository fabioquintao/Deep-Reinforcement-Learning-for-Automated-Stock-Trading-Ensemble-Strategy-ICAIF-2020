"""
JPMorgan Securities Services — Client Performance Workflow Monitor
==================================================================
Consumes workflow events from a FactSet-managed Kafka EMI topic,
transforms them, and indexes into Elasticsearch for observability.

Dependencies:
    pip install confluent-kafka elasticsearch loguru python-dotenv
"""

import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ElasticsearchException
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "jpm-perf-workflow-monitor")
    topic: str = os.getenv("KAFKA_TOPIC", "factset.emi.jpm.securities.performance")
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    session_timeout_ms: int = 30_000
    max_poll_interval_ms: int = 300_000
    # TLS / SASL for FactSet managed Kafka
    security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    sasl_mechanism: str = os.getenv("KAFKA_SASL_MECHANISM", "")
    sasl_username: str = os.getenv("KAFKA_SASL_USERNAME", "")
    sasl_password: str = os.getenv("KAFKA_SASL_PASSWORD", "")
    ssl_ca_location: str = os.getenv("KAFKA_SSL_CA_LOCATION", "")


@dataclass
class ElasticsearchConfig:
    hosts: list[str] = field(
        default_factory=lambda: os.getenv(
            "ES_HOSTS", "http://localhost:9200"
        ).split(",")
    )
    index_prefix: str = os.getenv("ES_INDEX_PREFIX", "jpm-perf-workflow")
    username: str = os.getenv("ES_USERNAME", "")
    password: str = os.getenv("ES_PASSWORD", "")
    ca_certs: str = os.getenv("ES_CA_CERTS", "")
    verify_certs: bool = os.getenv("ES_VERIFY_CERTS", "true").lower() == "true"
    bulk_batch_size: int = int(os.getenv("ES_BULK_BATCH_SIZE", "100"))
    bulk_flush_interval_seconds: int = int(os.getenv("ES_FLUSH_INTERVAL_SEC", "5"))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PerformanceEvent:
    """Normalised view of a JPMorgan client performance workflow event."""

    event_id: str
    client_id: str
    client_name: str
    workflow_name: str
    workflow_stage: str          # e.g. INITIATED, PROCESSING, COMPLETED, FAILED
    status: str                  # SUCCESS | FAILURE | PENDING
    portfolio_id: str
    benchmark_id: str
    period_start: str            # ISO-8601
    period_end: str              # ISO-8601
    return_pct: float | None
    benchmark_return_pct: float | None
    active_return_pct: float | None
    processing_latency_ms: int | None
    error_code: str | None
    error_message: str | None
    source_system: str           # e.g. "FACTSET_EMI"
    kafka_topic: str
    kafka_partition: int
    kafka_offset: int
    ingested_at: str             # ISO-8601 UTC timestamp added at ingest


def _parse_event(raw: dict[str, Any], meta: dict[str, Any]) -> PerformanceEvent:
    """Map a raw Kafka payload dict to a PerformanceEvent."""
    perf = raw.get("performance", {})
    workflow = raw.get("workflow", {})
    error = raw.get("error", {})

    return_pct = perf.get("return_pct")
    benchmark_return_pct = perf.get("benchmark_return_pct")
    active_return_pct = (
        round(return_pct - benchmark_return_pct, 6)
        if return_pct is not None and benchmark_return_pct is not None
        else perf.get("active_return_pct")
    )

    return PerformanceEvent(
        event_id=raw.get("event_id", ""),
        client_id=raw.get("client_id", ""),
        client_name=raw.get("client_name", ""),
        workflow_name=workflow.get("name", ""),
        workflow_stage=workflow.get("stage", ""),
        status=raw.get("status", ""),
        portfolio_id=raw.get("portfolio_id", ""),
        benchmark_id=raw.get("benchmark_id", ""),
        period_start=perf.get("period_start", ""),
        period_end=perf.get("period_end", ""),
        return_pct=return_pct,
        benchmark_return_pct=benchmark_return_pct,
        active_return_pct=active_return_pct,
        processing_latency_ms=workflow.get("latency_ms"),
        error_code=error.get("code"),
        error_message=error.get("message"),
        source_system="FACTSET_EMI",
        kafka_topic=meta["topic"],
        kafka_partition=meta["partition"],
        kafka_offset=meta["offset"],
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

INDEX_MAPPINGS = {
    "mappings": {
        "properties": {
            "event_id":               {"type": "keyword"},
            "client_id":              {"type": "keyword"},
            "client_name":            {"type": "keyword"},
            "workflow_name":          {"type": "keyword"},
            "workflow_stage":         {"type": "keyword"},
            "status":                 {"type": "keyword"},
            "portfolio_id":           {"type": "keyword"},
            "benchmark_id":           {"type": "keyword"},
            "period_start":           {"type": "date"},
            "period_end":             {"type": "date"},
            "return_pct":             {"type": "double"},
            "benchmark_return_pct":   {"type": "double"},
            "active_return_pct":      {"type": "double"},
            "processing_latency_ms":  {"type": "long"},
            "error_code":             {"type": "keyword"},
            "error_message":          {"type": "text"},
            "source_system":          {"type": "keyword"},
            "kafka_topic":            {"type": "keyword"},
            "kafka_partition":        {"type": "integer"},
            "kafka_offset":           {"type": "long"},
            "ingested_at":            {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
    },
}


def _index_name(prefix: str) -> str:
    """Daily rolling index, e.g. jpm-perf-workflow-2026.04.03"""
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y.%m.%d')}"


def build_es_client(cfg: ElasticsearchConfig) -> Elasticsearch:
    kwargs: dict[str, Any] = {"hosts": cfg.hosts}
    if cfg.username and cfg.password:
        kwargs["basic_auth"] = (cfg.username, cfg.password)
    if cfg.ca_certs:
        kwargs["ca_certs"] = cfg.ca_certs
        kwargs["verify_certs"] = cfg.verify_certs
    client = Elasticsearch(**kwargs)
    if not client.ping():
        raise ConnectionError(f"Cannot reach Elasticsearch at {cfg.hosts}")
    logger.info("Connected to Elasticsearch: {}", cfg.hosts)
    return client


def ensure_index(es: Elasticsearch, index: str) -> None:
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=INDEX_MAPPINGS)
        logger.info("Created index: {}", index)


def bulk_index(
    es: Elasticsearch,
    events: list[PerformanceEvent],
    index: str,
) -> tuple[int, int]:
    """Bulk-index events; returns (success_count, error_count)."""
    actions = [
        {
            "_index": index,
            "_id": e.event_id or None,
            "_source": asdict(e),
        }
        for e in events
    ]
    success, errors = helpers.bulk(es, actions, raise_on_error=False, stats_only=False)
    error_count = len(errors)
    if error_count:
        logger.warning("Bulk index errors ({} items): {}", error_count, errors[:3])
    return success, error_count


# ---------------------------------------------------------------------------
# Kafka consumer helpers
# ---------------------------------------------------------------------------

def build_kafka_consumer(cfg: KafkaConfig) -> Consumer:
    conf: dict[str, Any] = {
        "bootstrap.servers": cfg.bootstrap_servers,
        "group.id": cfg.group_id,
        "auto.offset.reset": cfg.auto_offset_reset,
        "enable.auto.commit": cfg.enable_auto_commit,
        "session.timeout.ms": cfg.session_timeout_ms,
        "max.poll.interval.ms": cfg.max_poll_interval_ms,
    }
    if cfg.security_protocol != "PLAINTEXT":
        conf["security.protocol"] = cfg.security_protocol
    if cfg.sasl_mechanism:
        conf["sasl.mechanism"] = cfg.sasl_mechanism
        conf["sasl.username"] = cfg.sasl_username
        conf["sasl.password"] = cfg.sasl_password
    if cfg.ssl_ca_location:
        conf["ssl.ca.location"] = cfg.ssl_ca_location

    consumer = Consumer(conf)
    consumer.subscribe([cfg.topic])
    logger.info("Subscribed to Kafka topic: {}", cfg.topic)
    return consumer


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class WorkflowMonitor:
    """
    Continuously polls Kafka for performance workflow events and
    writes them to Elasticsearch in micro-batches.
    """

    def __init__(
        self,
        kafka_cfg: KafkaConfig,
        es_cfg: ElasticsearchConfig,
    ) -> None:
        self._kafka_cfg = kafka_cfg
        self._es_cfg = es_cfg
        self._consumer: Consumer | None = None
        self._es: Elasticsearch | None = None
        self._running = False
        self._pending: list[PerformanceEvent] = []
        self._last_flush = time.monotonic()

        # stats
        self._total_consumed = 0
        self._total_indexed = 0
        self._total_errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._es = build_es_client(self._es_cfg)
        self._consumer = build_kafka_consumer(self._kafka_cfg)
        self._running = True

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Monitor started — waiting for events …")
        try:
            self._poll_loop()
        finally:
            self._shutdown()

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        logger.info("Shutdown signal received ({})", signum)
        self._running = False

    def _shutdown(self) -> None:
        logger.info("Flushing remaining {} events …", len(self._pending))
        self._flush()
        if self._consumer:
            self._consumer.close()
        if self._es:
            self._es.close()
        logger.info(
            "Monitor stopped. Consumed={} Indexed={} Errors={}",
            self._total_consumed,
            self._total_indexed,
            self._total_errors,
        )

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            msg = self._consumer.poll(timeout=1.0)

            if msg is None:
                self._maybe_flush()
                continue

            if msg.error():
                self._handle_kafka_error(msg)
                continue

            self._process_message(msg)
            self._maybe_flush()

    def _handle_kafka_error(self, msg: Any) -> None:
        err = msg.error()
        if err.code() == KafkaError._PARTITION_EOF:
            logger.debug(
                "End of partition {}/{} @ offset {}",
                msg.topic(), msg.partition(), msg.offset(),
            )
        elif err.fatal():
            raise KafkaException(err)
        else:
            logger.warning("Kafka error: {}", err)

    def _process_message(self, msg: Any) -> None:
        self._total_consumed += 1
        try:
            raw = json.loads(msg.value().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error(
                "Failed to decode message at {}/{}: {}",
                msg.partition(), msg.offset(), exc,
            )
            self._commit(msg)
            return

        meta = {
            "topic": msg.topic(),
            "partition": msg.partition(),
            "offset": msg.offset(),
        }
        try:
            event = _parse_event(raw, meta)
        except Exception as exc:
            logger.error("Failed to parse event: {} | raw={}", exc, raw)
            self._commit(msg)
            return

        self._pending.append(event)

        if len(self._pending) >= self._es_cfg.bulk_batch_size:
            self._flush()

        self._commit(msg)

    def _commit(self, msg: Any) -> None:
        self._consumer.commit(message=msg, asynchronous=True)

    # ------------------------------------------------------------------
    # Flush to Elasticsearch
    # ------------------------------------------------------------------

    def _maybe_flush(self) -> None:
        elapsed = time.monotonic() - self._last_flush
        if self._pending and elapsed >= self._es_cfg.bulk_flush_interval_seconds:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        index = _index_name(self._es_cfg.index_prefix)
        try:
            ensure_index(self._es, index)
            success, errors = bulk_index(self._es, self._pending, index)
            self._total_indexed += success
            self._total_errors += errors
            logger.info(
                "Flushed {} events to {} (success={} errors={})",
                len(self._pending), index, success, errors,
            )
        except ElasticsearchException as exc:
            logger.error("Elasticsearch flush failed: {}", exc)
            self._total_errors += len(self._pending)
        finally:
            self._pending.clear()
            self._last_flush = time.monotonic()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("LOG_LEVEL", "INFO"),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        "monitoring/logs/kafka_emi_monitor_{time}.log",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        level="DEBUG",
    )

    monitor = WorkflowMonitor(
        kafka_cfg=KafkaConfig(),
        es_cfg=ElasticsearchConfig(),
    )
    monitor.start()


if __name__ == "__main__":
    main()
