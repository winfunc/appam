//! Vision and document processing utilities.
//!
//! Helpers for working with images and documents in the Anthropic API.
//!
//! # Supported Formats
//!
//! ## Images
//! - JPEG (image/jpeg)
//! - PNG (image/png)
//! - GIF (image/gif)
//! - WebP (image/webp)
//!
//! ## Documents
//! - PDF (application/pdf)
//! - Plain text (text/plain)
//!
//! # Limits
//!
//! - Max image size: 5MB
//! - Max images per request: 100
//! - Max image dimensions: 8000x8000 px (or 2000x2000 px if >20 images)
//! - Optimal size: ≤1.15 megapixels, ≤1568px per dimension

use base64::{engine::general_purpose::STANDARD, Engine};

use super::types::{DocumentSource, ImageSource};

/// Encode image bytes to base64.
pub fn encode_image_base64(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}

/// Create a base64 image source.
pub fn image_source_base64(media_type: impl Into<String>, data: impl Into<String>) -> ImageSource {
    ImageSource::Base64 {
        media_type: media_type.into(),
        data: data.into(),
    }
}

/// Create a URL image source.
pub fn image_source_url(url: impl Into<String>) -> ImageSource {
    ImageSource::Url { url: url.into() }
}

/// Create a base64 PDF document source.
pub fn document_source_pdf_base64(data: impl Into<String>) -> DocumentSource {
    DocumentSource::Base64 {
        media_type: "application/pdf".to_string(),
        data: data.into(),
    }
}

/// Create a URL PDF document source.
pub fn document_source_pdf_url(url: impl Into<String>) -> DocumentSource {
    DocumentSource::Url { url: url.into() }
}

/// Create a plain text document source.
pub fn document_source_text(data: impl Into<String>) -> DocumentSource {
    DocumentSource::Text {
        media_type: "text/plain".to_string(),
        data: data.into(),
    }
}

/// Estimate token count for an image.
///
/// Formula: `(width * height) / 750`
///
/// This is an approximation. Images larger than optimal size may be resized.
pub fn estimate_image_tokens(width: u32, height: u32) -> u32 {
    (width * height) / 750
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_source_creation() {
        let base64_src = image_source_base64("image/jpeg", "base64data");
        match base64_src {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/jpeg");
                assert_eq!(data, "base64data");
            }
            _ => panic!("Expected base64 source"),
        }

        let url_src = image_source_url("https://example.com/image.jpg");
        match url_src {
            ImageSource::Url { url } => {
                assert_eq!(url, "https://example.com/image.jpg");
            }
            _ => panic!("Expected URL source"),
        }
    }

    #[test]
    fn test_estimate_image_tokens() {
        // 1092x1092 px (1:1 aspect ratio, optimal size)
        let tokens = estimate_image_tokens(1092, 1092);
        assert_eq!(tokens, 1589); // Approximately 1600 tokens

        // Small image
        let tokens = estimate_image_tokens(200, 200);
        assert_eq!(tokens, 53); // ~54 tokens
    }
}
