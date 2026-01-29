//
//  ModelManager.swift
//  chatgpt2
//
//  Singleton manager for GPT-2 model to avoid reloading on each ViewModel creation.
//

import Foundation

/// Manages the GPT-2 model instance as a singleton to avoid repeated loading
class ModelManager {
    /// Shared singleton instance
    static let shared = ModelManager()

    /// The GPT-2 model instance (lazy loaded)
    private(set) var model: GPT2?

    /// Error that occurred during model loading, if any
    private(set) var loadError: GPT2Error?

    /// Whether the model has been successfully loaded
    var isModelLoaded: Bool {
        return model != nil
    }

    private init() {
        loadModel()
    }

    /// Load the model with the default strategy
    private func loadModel() {
        do {
            model = try GPT2(strategy: .topK(GPT2Configuration.defaultTopK))
            loadError = nil
            print("ModelManager: GPT-2 model loaded successfully")
        } catch let error as GPT2Error {
            loadError = error
            model = nil
            print("ModelManager: Failed to load model - \(error.localizedDescription)")
        } catch {
            loadError = .modelInitializationFailed(error.localizedDescription)
            model = nil
            print("ModelManager: Failed to load model - \(error.localizedDescription)")
        }
    }

    /// Reload the model (useful after memory warnings or errors)
    func reloadModel() {
        model = nil
        loadModel()
    }

    /// Clear the model from memory (useful for memory warnings)
    func clearModel() {
        model = nil
        print("ModelManager: Model cleared from memory")
    }
}
