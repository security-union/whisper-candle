//! Output writers: txt / srt / vtt / tsv / json. Port of `whisper/utils.py`.

use crate::transcribe::TranscribeResult;
use crate::utils::format_timestamp;
use anyhow::Result;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Txt,
    Srt,
    Vtt,
    Tsv,
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "txt" => Self::Txt,
            "srt" => Self::Srt,
            "vtt" => Self::Vtt,
            "tsv" => Self::Tsv,
            "json" => Self::Json,
            _ => anyhow::bail!("unknown output format {s} (txt|srt|vtt|tsv|json|all)"),
        })
    }
}

impl OutputFormat {
    pub const ALL: [OutputFormat; 5] = [Self::Txt, Self::Srt, Self::Vtt, Self::Tsv, Self::Json];

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Tsv => "tsv",
            Self::Json => "json",
        }
    }

    pub fn render(&self, result: &TranscribeResult) -> Result<String> {
        let mut out = String::new();
        match self {
            Self::Txt => {
                for s in &result.segments {
                    if !s.text.is_empty() {
                        writeln!(out, "{}", s.text.trim())?;
                    }
                }
            }
            Self::Srt => {
                for (i, s) in result.segments.iter().enumerate() {
                    writeln!(
                        out,
                        "{}\n{} --> {}\n{}\n",
                        i + 1,
                        format_timestamp(s.start, true, ","),
                        format_timestamp(s.end, true, ","),
                        s.text.trim()
                    )?;
                }
            }
            Self::Vtt => {
                writeln!(out, "WEBVTT\n")?;
                for s in &result.segments {
                    writeln!(
                        out,
                        "{} --> {}\n{}\n",
                        format_timestamp(s.start, false, "."),
                        format_timestamp(s.end, false, "."),
                        s.text.trim()
                    )?;
                }
            }
            Self::Tsv => {
                writeln!(out, "start\tend\ttext")?;
                for s in &result.segments {
                    writeln!(
                        out,
                        "{}\t{}\t{}",
                        (s.start * 1000.0).round() as i64,
                        (s.end * 1000.0).round() as i64,
                        s.text.trim()
                    )?;
                }
            }
            Self::Json => {
                out = serde_json::to_string_pretty(result)?;
            }
        }
        Ok(out)
    }

    /// Write `<output_dir>/<audio_stem>.<ext>`.
    pub fn write_for(
        &self,
        result: &TranscribeResult,
        audio_path: &Path,
        output_dir: &Path,
    ) -> Result<std::path::PathBuf> {
        let stem = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let path = output_dir.join(format!("{stem}.{}", self.extension()));
        std::fs::write(&path, self.render(result)?)?;
        Ok(path)
    }
}
