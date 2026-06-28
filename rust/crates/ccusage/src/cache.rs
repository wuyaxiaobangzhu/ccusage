use std::{
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};

use crate::{
    LoadedEntry, LoadedFile, PricingMap, Speed, TimestampMs, TokenUsageRaw, UsageEntry,
    UsageMessage,
    calculate_cost_for_usage,
    cli::CostMode,
    missing_pricing_model_for_usage,
};

const CACHE_VERSION: u32 = 1;

/// Cached representation of a single usage entry from a Claude JSONL file.
///
/// Stores everything needed to reconstruct [`LoadedEntry`] except `cost` and
/// `missing_pricing_model`, which depend on the current pricing table and cost
/// mode and are recomputed on load (cheap arithmetic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CachedUsageEntry {
    // -- UsageEntry fields --
    session_id: Option<String>,
    timestamp: String,
    version: Option<String>,
    cost_usd: Option<f64>,
    request_id: Option<String>,
    is_api_error_message: Option<bool>,
    is_sidechain: Option<bool>,
    // -- TokenUsageRaw fields --
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
    speed: Option<Speed>,
    cache_creation_5m: u64,
    cache_creation_1h: u64,
    // -- UsageMessage fields --
    model: Option<String>,
    message_id: Option<String>,
    // -- Pre-computed fields (avoid re-parsing / re-deriving from path) --
    timestamp_ms: i64,
    date: String,
    project: String,
    session_id_derived: String,
    project_path: String,
    resolved_model: Option<String>,
    usage_limit_reset_time: Option<i64>,
}

impl CachedUsageEntry {
    /// Build a [`CachedUsageEntry`] from a fully-loaded [`LoadedEntry`].
    pub(crate) fn from_loaded(entry: &LoadedEntry) -> Self {
        let usage = entry.data.message.usage;
        let (cache_creation_5m, cache_creation_1h) = if let Some(breakdown) = usage.cache_creation {
            (breakdown.ephemeral_5m_input_tokens, breakdown.ephemeral_1h_input_tokens)
        } else {
            (usage.cache_creation_input_tokens, 0)
        };
        Self {
            session_id: entry.data.session_id.clone(),
            timestamp: entry.data.timestamp.clone(),
            version: entry.data.version.clone(),
            cost_usd: entry.data.cost_usd,
            request_id: entry.data.request_id.clone(),
            is_api_error_message: entry.data.is_api_error_message,
            is_sidechain: entry.data.is_sidechain,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            speed: usage.speed,
            cache_creation_5m,
            cache_creation_1h,
            model: entry.data.message.model.clone(),
            message_id: entry.data.message.id.clone(),
            timestamp_ms: entry.timestamp.as_millis(),
            date: entry.date.clone(),
            project: entry.project.to_string(),
            session_id_derived: entry.session_id.to_string(),
            project_path: entry.project_path.to_string(),
            resolved_model: entry.model.clone(),
            usage_limit_reset_time: entry.usage_limit_reset_time.map(|ts| ts.as_millis()),
        }
    }

    /// Reconstruct a [`LoadedEntry`] from this cached entry, recomputing cost
    /// and missing-pricing fields based on the current pricing/mode.
    pub(crate) fn to_loaded(
        &self,
        mode: CostMode,
        pricing: Option<&PricingMap>,
    ) -> LoadedEntry {
        let usage = TokenUsageRaw {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens,
            speed: self.speed,
            cache_creation: if self.cache_creation_5m > 0 || self.cache_creation_1h > 0 {
                Some(crate::types::CacheCreationRaw {
                    ephemeral_5m_input_tokens: self.cache_creation_5m,
                    ephemeral_1h_input_tokens: self.cache_creation_1h,
                })
            } else if self.cache_creation_input_tokens > 0 {
                // No breakdown was stored; keep flat representation
                None
            } else {
                None
            },
        };
        let data = UsageEntry {
            session_id: self.session_id.clone(),
            timestamp: self.timestamp.clone(),
            version: self.version.clone(),
            message: UsageMessage {
                usage,
                model: self.model.clone(),
                id: self.message_id.clone(),
            },
            cost_usd: self.cost_usd,
            request_id: self.request_id.clone(),
            is_api_error_message: self.is_api_error_message,
            is_sidechain: self.is_sidechain,
        };
        let cost = calculate_cost_for_usage(
            self.model.as_deref(),
            data.message.usage,
            self.cost_usd,
            mode,
            pricing,
        );
        let missing_pricing_model = missing_pricing_model_for_usage(
            self.model.as_deref(),
            data.message.usage,
            self.cost_usd,
            mode,
            pricing,
        );
        LoadedEntry {
            timestamp: TimestampMs::from_millis(self.timestamp_ms),
            date: self.date.clone(),
            project: Arc::from(self.project.as_str()),
            session_id: Arc::from(self.session_id_derived.as_str()),
            project_path: Arc::from(self.project_path.as_str()),
            cost,
            extra_total_tokens: 0,
            credits: None,
            message_count: None,
            model: self.resolved_model.clone(),
            usage_limit_reset_time: self.usage_limit_reset_time.map(TimestampMs::from_millis),
            missing_pricing_model,
            data,
        }
    }
}

/// Cached representation of a single entry from the daily summaries path.
///
/// The daily loader produces a slimmer entry than [`LoadedEntry`] and handles
/// `DailyAgentProgressLine` differently. This struct stores exactly the fields
/// needed to reconstruct that daily entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CachedDailyEntry {
    pub(crate) date: String,
    pub(crate) project: String,
    // -- TokenUsageRaw fields --
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cache_creation_input_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
    pub(crate) speed: Option<Speed>,
    pub(crate) cache_creation_5m: u64,
    pub(crate) cache_creation_1h: u64,
    pub(crate) cost: f64,
    pub(crate) model: Option<String>,
    pub(crate) missing_pricing_model: Option<String>,
    pub(crate) message_id: Option<String>,
    pub(crate) request_id: Option<String>,
    pub(crate) is_sidechain: Option<bool>,
}

impl CachedDailyEntry {
    /// Build from the individual fields of a daily loaded entry.
    pub(crate) fn from_daily_fields(
        date: String,
        project: &str,
        usage: crate::TokenUsageRaw,
        cost: f64,
        model: Option<String>,
        missing_pricing_model: Option<String>,
        message_id: Option<String>,
        request_id: Option<String>,
        is_sidechain: Option<bool>,
    ) -> Self {
        let (cache_creation_5m, cache_creation_1h) =
            if let Some(breakdown) = usage.cache_creation {
                (breakdown.ephemeral_5m_input_tokens, breakdown.ephemeral_1h_input_tokens)
            } else {
                (usage.cache_creation_input_tokens, 0)
            };
        Self {
            date,
            project: project.to_string(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            speed: usage.speed,
            cache_creation_5m,
            cache_creation_1h,
            cost,
            model,
            missing_pricing_model,
            message_id,
            request_id,
            is_sidechain,
        }
    }

    /// Reconstruct a `TokenUsageRaw` from this cached entry.
    pub(crate) fn to_token_usage(&self) -> crate::TokenUsageRaw {
        crate::TokenUsageRaw {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens,
            speed: self.speed,
            cache_creation: if self.cache_creation_5m > 0 || self.cache_creation_1h > 0 {
                Some(crate::types::CacheCreationRaw {
                    ephemeral_5m_input_tokens: self.cache_creation_5m,
                    ephemeral_1h_input_tokens: self.cache_creation_1h,
                })
            } else {
                None
            },
        }
    }
}

/// On-disk cache for a single source JSONL file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FileCache<T> {
    version: u32,
    source_mtime_secs: u64,
    source_size: u64,
    pub(crate) entries: Vec<T>,
}

/// Resolve the cache directory, creating it if needed.
fn cache_dir() -> PathBuf {
    let base = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            crate::home::home_dir().map(|home| home.join(".cache"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("ccusage-cache"));
    base.join("ccusage").join("v1")
}

/// Derive a cache file path from the source file's absolute path.
fn cache_path_for(source: &Path) -> PathBuf {
    let mut hasher = FxHasher::default();
    // Hash the path bytes directly. The source paths from `usage_files` are
    // already absolute, so canonicalization is unnecessary overhead.
    source.as_os_str().hash(&mut hasher);
    cache_dir().join(format!("{:016x}.msgpack", hasher.finish()))
}

/// Read the source file's mtime (seconds since epoch) and size.
fn source_mtime_and_size(path: &Path) -> Option<(u64, u64)> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some((mtime, meta.len()))
}

/// Try to read and validate the cache for `source`. Returns `None` on any
/// failure (missing file, corrupt data, mtime/size mismatch, version mismatch).
pub(crate) fn read_file_cache<T: serde::de::DeserializeOwned>(
    source: &Path,
) -> Option<FileCache<T>> {
    let (mtime, size) = source_mtime_and_size(source)?;
    let cache_path = cache_path_for(source);
    let bytes = fs::read(&cache_path).ok()?;
    let cache: FileCache<T> = rmp_serde::from_slice(&bytes).ok()?;
    if cache.version != CACHE_VERSION {
        return None;
    }
    if cache.source_mtime_secs != mtime || cache.source_size != size {
        return None;
    }
    Some(cache)
}

/// Write the cache for `source`. Failures are silently ignored (cache is
/// best-effort).
pub(crate) fn write_file_cache<T: Serialize + Clone>(source: &Path, entries: &[T]) {
    let Some((mtime, size)) = source_mtime_and_size(source) else {
        return;
    };
    let cache = FileCache {
        version: CACHE_VERSION,
        source_mtime_secs: mtime,
        source_size: size,
        entries: entries.to_vec(),
    };
    let Ok(bytes) = rmp_serde::to_vec(&cache) else {
        return;
    };
    let cache_path = cache_path_for(source);
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    // Write to a temp file then rename for atomicity.
    let tmp_path = cache_path.with_extension("msgpack.tmp");
    if fs::write(&tmp_path, &bytes).is_ok() {
        let _ = fs::rename(&tmp_path, &cache_path);
    }
}

/// Try to load a source file from cache. Returns `Some(LoadedFile)` with
/// reconstructed entries on cache hit. On miss returns `None` so the caller
/// falls back to normal parsing.
pub(crate) fn try_load_cached(
    source: &Path,
    mode: CostMode,
    pricing: Option<&PricingMap>,
) -> Option<LoadedFile> {
    let cache: FileCache<CachedUsageEntry> = read_file_cache(source)?;
    let entries: Vec<LoadedEntry> = cache
        .entries
        .into_iter()
        .map(|cached| cached.to_loaded(mode, pricing))
        .collect();
    // Recover the file-level timestamp from the earliest entry.
    let timestamp = entries
        .iter()
        .map(|entry| entry.timestamp)
        .min();
    Some(LoadedFile { timestamp, entries })
}

/// Build cached entries from a freshly-parsed `LoadedFile` and write the cache.
pub(crate) fn write_loaded_file_cache(source: &Path, loaded: &LoadedFile) {
    let cached: Vec<CachedUsageEntry> = loaded
        .entries
        .iter()
        .map(CachedUsageEntry::from_loaded)
        .collect();
    write_file_cache(source, &cached);
}

/// Try to read the daily cache for a source file. Returns the raw cached
/// entries on hit, or `None` on miss.
pub(crate) fn read_daily_file_cache(source: &Path) -> Option<FileCache<CachedDailyEntry>> {
    read_file_cache(source)
}

/// Write daily cache entries for a source file.
pub(crate) fn write_daily_file_cache(source: &Path, entries: &[CachedDailyEntry]) {
    write_file_cache(source, entries);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        TokenUsageRaw, UsageEntry, UsageMessage,
        cli::CostMode,
    };
    use ccusage_test_support::Fixture;

    fn sample_loaded_entry() -> LoadedEntry {
        LoadedEntry {
            data: UsageEntry {
                session_id: Some("session-abc".to_string()),
                timestamp: "2026-06-28T12:00:00.000Z".to_string(),
                version: Some("1.0.0".to_string()),
                message: UsageMessage {
                    usage: TokenUsageRaw {
                        input_tokens: 100,
                        output_tokens: 50,
                        cache_creation_input_tokens: 10,
                        cache_read_input_tokens: 80,
                        speed: None,
                        cache_creation: None,
                    },
                    model: Some("claude-sonnet-4-20250514".to_string()),
                    id: Some("msg-001".to_string()),
                },
                cost_usd: Some(0.001),
                request_id: Some("req-001".to_string()),
                is_api_error_message: None,
                is_sidechain: None,
            },
            timestamp: TimestampMs::from_millis(1_751_112_000_000),
            date: "2026-06-28".to_string(),
            project: Arc::from("my-project"),
            session_id: Arc::from("session-abc"),
            project_path: Arc::from("my-project"),
            cost: 0.001,
            extra_total_tokens: 0,
            credits: None,
            message_count: None,
            model: Some("claude-sonnet-4-20250514".to_string()),
            usage_limit_reset_time: None,
            missing_pricing_model: None,
        }
    }

    #[test]
    fn roundtrip_cached_usage_entry() {
        let entry = sample_loaded_entry();
        let cached = CachedUsageEntry::from_loaded(&entry);
        let restored = cached.to_loaded(CostMode::Display, None);

        assert_eq!(restored.data.session_id, entry.data.session_id);
        assert_eq!(restored.data.timestamp, entry.data.timestamp);
        assert_eq!(restored.data.message.usage.input_tokens, 100);
        assert_eq!(restored.data.message.usage.output_tokens, 50);
        assert_eq!(restored.data.message.usage.cache_read_input_tokens, 80);
        assert_eq!(restored.date, "2026-06-28");
        assert_eq!(restored.project.as_ref(), "my-project");
        assert_eq!(restored.session_id.as_ref(), "session-abc");
        // Display mode uses cost_usd from the JSONL
        assert!((restored.cost - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn file_cache_roundtrip_via_messagepack() {
        let entry = CachedUsageEntry::from_loaded(&sample_loaded_entry());
        let cache = FileCache {
            version: CACHE_VERSION,
            source_mtime_secs: 12345,
            source_size: 6789,
            entries: vec![entry],
        };
        let bytes = rmp_serde::to_vec(&cache).unwrap();
        let restored: FileCache<CachedUsageEntry> = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(restored.version, CACHE_VERSION);
        assert_eq!(restored.source_mtime_secs, 12345);
        assert_eq!(restored.entries.len(), 1);
        assert_eq!(restored.entries[0].input_tokens, 100);
    }

    #[test]
    fn cache_hit_when_source_unchanged() {
        let fixture = Fixture::new();
        let source = fixture.write_file("session.jsonl", r#"{"test":true}"#);

        let entries = vec![CachedUsageEntry::from_loaded(&sample_loaded_entry())];
        write_file_cache(&source, &entries);

        let cache: Option<FileCache<CachedUsageEntry>> = read_file_cache(&source);
        assert!(cache.is_some());
        assert_eq!(cache.unwrap().entries.len(), 1);
    }

    #[test]
    fn cache_miss_when_source_modified() {
        let fixture = Fixture::new();
        let source = fixture.write_file("session.jsonl", r#"{"test":true}"#);

        let entries = vec![CachedUsageEntry::from_loaded(&sample_loaded_entry())];
        write_file_cache(&source, &entries);

        // Touch the file to change mtime
        std::thread::sleep(std::time::Duration::from_millis(50));
        fs::write(&source, r#"{"test":true,"extra":1}"#).unwrap();

        let cache: Option<FileCache<CachedUsageEntry>> = read_file_cache(&source);
        assert!(cache.is_none());
    }

    #[test]
    fn cache_miss_on_corrupt_file() {
        let fixture = Fixture::new();
        let source = fixture.write_file("session.jsonl", r#"{"test":true}"#);

        // Write cache with valid content, then corrupt it
        let entries = vec![CachedUsageEntry::from_loaded(&sample_loaded_entry())];
        write_file_cache(&source, &entries);
        let cache_path = cache_path_for(&source);
        fs::write(&cache_path, b"corrupt data").unwrap();

        let cache: Option<FileCache<CachedUsageEntry>> = read_file_cache(&source);
        assert!(cache.is_none());
    }

    #[test]
    fn cache_miss_when_source_missing() {
        let cache: Option<FileCache<CachedUsageEntry>> =
            read_file_cache(Path::new("/nonexistent/path/session.jsonl"));
        assert!(cache.is_none());
    }
}
