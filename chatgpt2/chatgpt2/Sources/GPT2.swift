//
//  GPT2.swift
//  CoreMLGPT2
//
//  Created by Julien Chaumond on 19/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import CoreML


class GPT2 {

    enum DecodingStrategy {
        /// At each time step, we select the most likely next token
        case greedy
        /// Sample only from the top-k most-probable tokens (k is a hyper-parameter).
        case topK(Int)

    }

    private let model: JointModel
    public let tokenizer: GPT2Tokenizer
    public let seqLen = GPT2Configuration.sequenceLength
    private let strategy: DecodingStrategy

    private var isStopped = false

    /// Last error that occurred during prediction
    private(set) var lastError: GPT2Error?

    init(strategy: DecodingStrategy = .greedy) throws {
        self.strategy = strategy
        do {
            self.model = try JointModel()
        } catch {
            throw GPT2Error.modelInitializationFailed(error.localizedDescription)
        }

        do {
            self.tokenizer = try GPT2Tokenizer()
        } catch {
            throw GPT2Error.tokenizationFailed(error.localizedDescription)
        }
    }

    func stopGeneration() {
        isStopped = true
    }

    /// Main prediction loop:
    /// Predict next token from array of previous tokens.
    /// - featurization
    /// - model inference
    /// - Decoding according to the model's `strategy`
    func predict(tokens: [Int]) -> Int? {
        lastError = nil

        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens

        /// Pad input_ids on the right, up to `seqLen`:
        let input_ids = MLMultiArray.from(
            maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count)
        )

        let output: JointModelOutput
        do {
            output = try model.prediction(input_ids: input_ids)
        } catch {
            lastError = .predictionFailed(error.localizedDescription)
            return nil
        }

        let outputLogits = MLMultiArray.slice(
            output.linear_0,
            indexing: [.select(maxTokens.count - 1), .slice]
        )

        switch strategy {
        case .greedy:
            let nextToken = Math.argmax(outputLogits)
            return nextToken.0
        case .topK(let k):
            let logits = MLMultiArray.toDoubleArray(outputLogits)
            let topk = Math.topK(arr: logits, k: k)
            let sampleIndex = Math.sample(indexes: topk.indexes, probs: topk.probs)
            return sampleIndex
        }
    }


    /// Main generation loop.
    ///
    /// Will generate next `nTokens` (defaults to 10).
    /// Calls an incremental `callback` for each new token, then returns the generated string at the end.
    ///
    func generate(text: String, nTokens: Int = GPT2Configuration.defaultGenerationTokens, callback: ((String, Double) -> Void)?) -> String {
        var tokens: [Int]
        do {
            tokens = try tokenizer.encode(text: text)
        } catch {
            lastError = .tokenizationFailed(error.localizedDescription)
            return ""
        }

        var newTokens: [Int] = []
        isStopped = false

        for i in 0..<nTokens {
            if isStopped { break }
            let (nextToken, time) = Utils.time {
                return predict(tokens: tokens)
            }

            guard let token = nextToken else {
                print("Prediction failed at token \(i)")
                break
            }

            if token == GPT2Configuration.endOfSequenceToken {
                print("EOS token detected, stopping generation.")
                break
            }
            tokens.append(token)
            newTokens.append(token)
            print(" <\(time)s>", i, token, tokens.count)

            do {
                let decoded = try tokenizer.decode(tokens: newTokens)
                callback?(decoded, time)
            } catch {
                print("Failed to decode tokens: \(error)")
            }
        }

        do {
            return try tokenizer.decode(tokens: newTokens)
        } catch {
            lastError = .tokenizationFailed(error.localizedDescription)
            return ""
        }
    }
}
