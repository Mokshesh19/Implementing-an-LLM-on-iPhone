//
//  GPT2Error.swift
//  chatgpt2
//
//  Custom error types for better error handling and debugging.
//

import Foundation

/// Errors that can occur during GPT-2 model operations
enum GPT2Error: LocalizedError {
    case modelInitializationFailed(String)
    case predictionFailed(String)
    case tokenizationFailed(String)
    case resourceNotFound(String)
    case invalidToken(Int)
    case invalidInput(String)
    case arrayCreationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelInitializationFailed(let reason):
            return "Failed to initialize model: \(reason)"
        case .predictionFailed(let reason):
            return "Prediction failed: \(reason)"
        case .tokenizationFailed(let reason):
            return "Tokenization failed: \(reason)"
        case .resourceNotFound(let resource):
            return "Resource not found: \(resource)"
        case .invalidToken(let token):
            return "Invalid token ID: \(token)"
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        case .arrayCreationFailed(let reason):
            return "Failed to create MLMultiArray: \(reason)"
        }
    }
}

/// Configuration constants for the GPT-2 model
struct GPT2Configuration {
    /// Maximum sequence length for input tokens
    static let sequenceLength = 64

    /// End of sequence token ID
    static let endOfSequenceToken = 50256

    /// Default top-k value for sampling
    static let defaultTopK = 40

    /// Vocabulary size for GPT-2
    static let vocabularySize = 50257

    /// Default number of tokens to generate
    static let defaultGenerationTokens = 64
}
