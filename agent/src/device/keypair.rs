use ed25519_dalek::{SigningKey, VerifyingKey};

/// Custom serde serialization for Ed25519 keypair using multibase Base58BTC.
///
/// This module provides serialization/deserialization for Ed25519 signing keys
/// using multibase Base58BTC encoding (with 'z' prefix). This ensures
/// compatibility with the reference implementation and P2P standards.
///
/// Pattern adapted from: /Users/deepsaint/Desktop/meshnet/reference/config.rs lines 12-37
pub mod keypair_serde {
    use super::*;
    use serde::{de, Deserialize, Deserializer, Serializer};

    /// Serialize SigningKey to multibase Base58BTC string.
    ///
    /// Format: 'z' prefix + Base58BTC encoded bytes
    pub fn serialize<S>(key: &SigningKey, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = multibase::encode(multibase::Base::Base58Btc, key.to_bytes());
        serializer.serialize_str(&encoded)
    }

    /// Deserialize SigningKey from multibase Base58BTC string.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<SigningKey, D::Error>
    where
        D: Deserializer<'de>,
    {
        let string = String::deserialize(deserializer)?;
        let (_, decoded) = multibase::decode(&string).map_err(de::Error::custom)?;

        // ed25519-dalek 2.x uses from_bytes instead of try_from_bytes
        let key_bytes: [u8; 32] = decoded
            .as_slice()
            .try_into()
            .map_err(|_| de::Error::custom("Invalid key length (expected 32 bytes)"))?;

        Ok(SigningKey::from_bytes(&key_bytes))
    }
}

/// Generate a new Ed25519 signing keypair.
///
/// Uses the OS random number generator for cryptographically secure randomness.
///
/// # Example
///
/// ```
/// use mesh_agent::device::keypair::generate_keypair;
///
/// let signing_key = generate_keypair();
/// let verifying_key = public_key(&signing_key);
/// ```
pub fn generate_keypair() -> SigningKey {
    let mut rng = rand::rngs::OsRng;
    // ed25519-dalek 2.x uses from_bytes with random bytes
    let mut secret_bytes = [0u8; 32];
    rand::RngCore::fill_bytes(&mut rng, &mut secret_bytes);
    SigningKey::from_bytes(&secret_bytes)
}

/// Extract the public (verifying) key from a signing key.
pub fn public_key(signing_key: &SigningKey) -> VerifyingKey {
    signing_key.verifying_key()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct TestStruct {
        #[serde(with = "keypair_serde")]
        key: SigningKey,
    }

    #[test]
    fn test_keypair_generation() {
        let key1 = generate_keypair();
        let key2 = generate_keypair();

        // Keys should be different
        assert_ne!(key1.to_bytes(), key2.to_bytes());
    }

    #[test]
    fn test_public_key_extraction() {
        let signing_key = generate_keypair();
        let verifying_key = public_key(&signing_key);

        // Verify they match
        assert_eq!(
            verifying_key.to_bytes(),
            signing_key.verifying_key().to_bytes()
        );
    }

    #[test]
    fn test_keypair_serialization_roundtrip() {
        let original = generate_keypair();

        let test_struct = TestStruct {
            key: original.clone(),
        };

        // Serialize
        let json = serde_json::to_string(&test_struct).unwrap();

        // Verify format starts with 'z' (multibase Base58BTC prefix)
        assert!(
            json.contains("\"z"),
            "Encoded key should start with 'z' prefix"
        );

        // Deserialize
        let deserialized: TestStruct = serde_json::from_str(&json).unwrap();

        // Verify bytes match exactly
        assert_eq!(
            original.to_bytes(),
            deserialized.key.to_bytes(),
            "Keypair bytes should be identical after roundtrip"
        );
    }

    #[test]
    fn test_multibase_base58btc_format() {
        let key = generate_keypair();
        let test_struct = TestStruct { key };

        let json = serde_json::to_string(&test_struct).unwrap();

        // Extract the encoded string
        let encoded_str: serde_json::Value = serde_json::from_str(&json).unwrap();
        let key_str = encoded_str["key"].as_str().unwrap();

        // Verify multibase Base58BTC format
        assert!(key_str.starts_with('z'), "Should start with 'z' prefix");

        // Verify decoding works
        let (base, decoded) = multibase::decode(key_str).unwrap();
        assert_eq!(base, multibase::Base::Base58Btc);
        assert_eq!(decoded.len(), 32, "Ed25519 key should be 32 bytes");
    }

    #[test]
    fn test_invalid_key_length_error() {
        // Create a valid multibase Base58BTC string but with wrong length (16 bytes instead of 32)
        let short_bytes = vec![0u8; 16];
        let invalid_key = multibase::encode(multibase::Base::Base58Btc, short_bytes);
        let invalid_json = format!(r#"{{"key":"{}"}}"#, invalid_key);

        let result: Result<TestStruct, _> = serde_json::from_str(&invalid_json);

        assert!(result.is_err(), "Should fail with invalid key length");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid key length") || err_msg.contains("expected 32 bytes"));
    }

    #[test]
    fn test_invalid_multibase_error() {
        let invalid_json = r#"{"key":"not-multibase-encoded"}"#;
        let result: Result<TestStruct, _> = serde_json::from_str(invalid_json);

        assert!(
            result.is_err(),
            "Should fail with invalid multibase encoding"
        );
    }

    #[test]
    fn test_keypair_bytes_length() {
        let key = generate_keypair();
        assert_eq!(key.to_bytes().len(), 32, "Ed25519 signing key is 32 bytes");

        let pub_key = public_key(&key);
        assert_eq!(
            pub_key.to_bytes().len(),
            32,
            "Ed25519 verifying key is 32 bytes"
        );
    }
}
