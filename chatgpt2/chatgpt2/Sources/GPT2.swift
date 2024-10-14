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
    
    private let model : JointModel
    public let tokenizer = GPT2Tokenizer()
    public let seqLen = 64
    private let strategy: DecodingStrategy
    
    private var isStopped = false
    
    init(strategy: DecodingStrategy = .greedy) {
            self.strategy = strategy
            do {
                self.model = try JointModel()
            } catch {
                fatalError("Failed to initialize the model: \(error)")
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
    func predict(tokens: [Int]) -> Int {
        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens
        
        /// Pad input_ids on the right, up to `seqLen`:
        let input_ids = MLMultiArray.from(
            maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count)
        )
        let _position_ids = MLMultiArray.from(
            Array(0..<seqLen)
        )
        
        let output = try! model.prediction(input_ids: input_ids)
        
//        let outputLogits = MLMultiArray.slice(
//            output.linear_0,
//            indexing: [.select(0), .select(maxTokens.count - 1), .slice, .select(0), .select(0)]
//        )
        
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
    func generate(text: String, nTokens: Int = 10, callback: ((String, Double) -> Void)?) -> String {
        var tokens = tokenizer.encode(text: text)
        var newTokens: [Int] = []
        isStopped = false

        for i in 0..<nTokens {
            if isStopped { break }
            let (nextToken, time) = Utils.time {
                
                return predict(tokens: tokens)
            }
            if nextToken == 50256 {  // Stop if EOS token is generated
                print("EOS token detected, stopping generation.")
                break
            }
            tokens.append(nextToken)
            newTokens.append(nextToken)
            print(" <\(time)s>", i, nextToken, tokens.count)
            callback?(
                tokenizer.decode(tokens: newTokens), time
            )
        }
        return tokenizer.decode(tokens: newTokens)
    }
}
