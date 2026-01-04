use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use time::OffsetDateTime;
use tracing::{debug, info};

/// Control plane's Ed25519 keypair for signing device certificates
pub struct ControlPlaneKeypair {
    signing_key: SigningKey,
    #[allow(dead_code)]
    keypair_path: PathBuf,
}

impl ControlPlaneKeypair {
    /// Load or generate control plane keypair
    pub fn load_or_generate() -> Result<Self, Box<dyn std::error::Error>> {
        let keypair_path = Self::default_path()?;

        if keypair_path.exists() {
            info!(path = %keypair_path.display(), "Loading control plane keypair");
            Self::load(&keypair_path)
        } else {
            info!(path = %keypair_path.display(), "Generating new control plane keypair");
            Self::generate(&keypair_path)
        }
    }

    /// Default path for control plane keypair
    fn default_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let meshnet_dir = dirs::home_dir()
            .ok_or("Failed to get home directory")?
            .join(".meshnet");

        // Create directory if it doesn't exist
        if !meshnet_dir.exists() {
            fs::create_dir_all(&meshnet_dir)?;
        }

        Ok(meshnet_dir.join("control-plane-key.toml"))
    }

    /// Generate new keypair and save to file
    fn generate(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let signing_key = SigningKey::generate(&mut OsRng);

        // Serialize keypair
        let keypair_data = KeypairData {
            secret_key: multibase::encode(multibase::Base::Base58Btc, signing_key.to_bytes()),
        };

        let toml_content = toml::to_string_pretty(&keypair_data)?;

        // Atomic write (temp file + rename)
        let temp_path = path.with_extension("tmp");
        fs::write(&temp_path, toml_content)?;
        fs::rename(&temp_path, path)?;

        debug!("Generated and saved control plane keypair");

        Ok(Self {
            signing_key,
            keypair_path: path.to_path_buf(),
        })
    }

    /// Load keypair from file
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let keypair_data: KeypairData = toml::from_str(&content)?;

        // Decode secret key from multibase
        let (_, decoded) = multibase::decode(&keypair_data.secret_key)?;
        if decoded.len() != 32 {
            return Err("Invalid secret key length".into());
        }

        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&decoded);
        let signing_key = SigningKey::from_bytes(&bytes);

        debug!("Loaded control plane keypair");

        Ok(Self {
            signing_key,
            keypair_path: path.to_path_buf(),
        })
    }

    /// Get the public verifying key
    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Generate a certificate for a device (MVP self-signed)
    pub fn generate_certificate(
        &self,
        device_id: &str,
        network_id: &str,
        public_key: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if public_key.len() != 32 {
            return Err("Public key must be 32 bytes".into());
        }

        let now = OffsetDateTime::now_utc();
        let expires_at = now + time::Duration::days(30); // 30 day validity for MVP

        // Create certificate structure
        let cert = DeviceCertificate {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            public_key: public_key.to_vec(),
            issued_at: now.format(&time::format_description::well_known::Rfc3339)?,
            expires_at: expires_at.format(&time::format_description::well_known::Rfc3339)?,
        };

        // Serialize certificate to CBOR
        let mut cert_bytes = Vec::new();
        ciborium::into_writer(&cert, &mut cert_bytes)?;

        // Sign the certificate bytes with control plane's private key
        use ed25519_dalek::Signer;
        let signature = self.signing_key.sign(&cert_bytes);

        // Create signed certificate
        let signed_cert = SignedDeviceCertificate {
            certificate: cert_bytes,
            signature: signature.to_bytes().to_vec(),
        };

        // Serialize the signed certificate to CBOR (final blob)
        let mut final_bytes = Vec::new();
        ciborium::into_writer(&signed_cert, &mut final_bytes)?;

        debug!(
            device_id = %device_id,
            network_id = %network_id,
            size = final_bytes.len(),
            "Generated device certificate"
        );

        Ok(final_bytes)
    }

    /// Verify a device certificate (for testing/validation)
    pub fn verify_certificate(
        &self,
        cert_blob: &[u8],
    ) -> Result<DeviceCertificate, Box<dyn std::error::Error>> {
        // Deserialize signed certificate
        let signed_cert: SignedDeviceCertificate = ciborium::from_reader(cert_blob)?;

        // Verify signature
        use ed25519_dalek::{Signature, Verifier};
        let signature_bytes: [u8; 64] = signed_cert
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| "Invalid signature length")?;
        let signature = Signature::from_bytes(&signature_bytes);

        self.verifying_key()
            .verify(&signed_cert.certificate, &signature)
            .map_err(|_| "Invalid certificate signature")?;

        // Deserialize certificate
        let cert: DeviceCertificate = ciborium::from_reader(&signed_cert.certificate[..])?;

        Ok(cert)
    }
}

/// Keypair data for TOML serialization
#[derive(Serialize, Deserialize)]
struct KeypairData {
    secret_key: String, // Multibase Base58BTC encoded
}

/// Device certificate structure (MVP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCertificate {
    pub device_id: String,
    pub network_id: String,
    pub public_key: Vec<u8>,
    pub issued_at: String,  // ISO 8601
    pub expires_at: String, // ISO 8601
}

/// Signed device certificate (certificate + signature)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignedDeviceCertificate {
    certificate: Vec<u8>, // CBOR-encoded DeviceCertificate
    signature: Vec<u8>,   // Ed25519 signature (64 bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_keypair_generation() {
        let temp_dir = env::temp_dir();
        let keypair_path = temp_dir.join("test_control_plane_key.toml");

        // Clean up if exists
        let _ = fs::remove_file(&keypair_path);

        // Generate new keypair
        let keypair = ControlPlaneKeypair::generate(&keypair_path).unwrap();
        assert!(keypair_path.exists());

        // Load the same keypair
        let loaded_keypair = ControlPlaneKeypair::load(&keypair_path).unwrap();
        assert_eq!(
            keypair.verifying_key().to_bytes(),
            loaded_keypair.verifying_key().to_bytes()
        );

        // Clean up
        fs::remove_file(&keypair_path).unwrap();
    }

    #[test]
    fn test_certificate_generation_and_verification() {
        let temp_dir = env::temp_dir();
        let keypair_path = temp_dir.join("test_cert_gen_key.toml");
        let _ = fs::remove_file(&keypair_path);

        let keypair = ControlPlaneKeypair::generate(&keypair_path).unwrap();

        // Generate certificate
        let device_id = "device-123";
        let network_id = "network-456";
        let public_key = [42u8; 32]; // Dummy public key

        let cert_blob = keypair
            .generate_certificate(device_id, network_id, &public_key)
            .unwrap();

        assert!(!cert_blob.is_empty());
        assert!(cert_blob.len() < 1000); // Should be reasonably compact (CBOR + signature)

        // Verify certificate
        let cert = keypair.verify_certificate(&cert_blob).unwrap();
        assert_eq!(cert.device_id, device_id);
        assert_eq!(cert.network_id, network_id);
        assert_eq!(cert.public_key, public_key);

        // Clean up
        fs::remove_file(&keypair_path).unwrap();
    }

    #[test]
    fn test_invalid_signature_fails() {
        let temp_dir = env::temp_dir();
        let keypair_path = temp_dir.join("test_invalid_sig_key.toml");
        let _ = fs::remove_file(&keypair_path);

        let keypair = ControlPlaneKeypair::generate(&keypair_path).unwrap();

        let cert_blob = keypair
            .generate_certificate("dev1", "net1", &[1u8; 32])
            .unwrap();

        // Tamper with certificate
        let mut tampered = cert_blob.clone();
        tampered[10] ^= 0xFF; // Flip some bits

        // Verification should fail
        assert!(keypair.verify_certificate(&tampered).is_err());

        // Clean up
        fs::remove_file(&keypair_path).unwrap();
    }

    #[test]
    fn test_invalid_public_key_length() {
        let temp_dir = env::temp_dir();
        let keypair_path = temp_dir.join("test_invalid_pk_key.toml");
        let _ = fs::remove_file(&keypair_path);

        let keypair = ControlPlaneKeypair::generate(&keypair_path).unwrap();

        // Public key with wrong length
        let result = keypair.generate_certificate("dev1", "net1", &[1u8; 16]);
        assert!(result.is_err());

        // Clean up
        fs::remove_file(&keypair_path).unwrap();
    }
}
