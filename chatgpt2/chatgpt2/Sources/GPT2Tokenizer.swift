//
//  GPT2Tokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation

struct BytePair: Hashable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }
    init(tuple: [String]) {
        self.a = tuple[0]
        self.b = tuple[1]
    }

    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        return lhs.a == rhs.a && lhs.b == rhs.b
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}

fileprivate extension String {
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start = range.lowerBound < range.upperBound ? range.upperBound : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }
}




class GPT2Tokenizer {
    let bpeRanks: Dictionary<BytePair, Int>
    private let encoder: [String: Int]
    private let decoder: [Int: String]

    init() throws {
        guard let mergesUrl = Bundle.main.url(forResource: "gpt2-merges", withExtension: "txt") else {
            throw GPT2Error.resourceNotFound("gpt2-merges.txt")
        }

        let bpeMergesTxt: String
        do {
            bpeMergesTxt = try String(contentsOf: mergesUrl)
        } catch {
            throw GPT2Error.resourceNotFound("Failed to read gpt2-merges.txt: \(error.localizedDescription)")
        }

        let arr = bpeMergesTxt.split(separator: "\n").map { String($0) }
        var bpeRanks: Dictionary<BytePair, Int> = [:]
        for i in 1..<arr.count {
            let tuple = arr[i].split(separator: " ").map { String($0) }
            let bp = BytePair(tuple: tuple)
            bpeRanks[bp] = i - 1
        }
        self.bpeRanks = bpeRanks

        self.encoder = try {
            guard let vocabUrl = Bundle.main.url(forResource: "gpt2-vocab", withExtension: "json") else {
                throw GPT2Error.resourceNotFound("gpt2-vocab.json")
            }

            let json: Data
            do {
                json = try Data(contentsOf: vocabUrl)
            } catch {
                throw GPT2Error.resourceNotFound("Failed to read gpt2-vocab.json: \(error.localizedDescription)")
            }

            let decoder = JSONDecoder()
            do {
                let vocab = try decoder.decode([String: Int].self, from: json)
                return vocab
            } catch {
                throw GPT2Error.tokenizationFailed("Failed to decode vocabulary: \(error.localizedDescription)")
            }
        }()
        self.decoder = Utils.invert(self.encoder)
    }

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.compactMap { (token) -> String? in
            let encoded = Array(token.utf8).compactMap { byteEncoder[$0] }
            guard encoded.count == Array(token.utf8).count else {
                return nil
            }
            return encoded.joined()
        }
    }

    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count-1 {
            let bp = BytePair(
                word[i],
                word[i+1]
            )
            s.insert(bp)
        }
        return s
    }

    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token
        }

        var word = Array(token).map { String($0) }
        var pairs = Array(getPairs(word: word))

        while true {
            let bigrams = pairs.filter { (bp) -> Bool in bpeRanks[bp] != nil }
            if bigrams.count == 0 {
                break
            }
            guard let bigram = bigrams.min(by: { (bp1, bp2) -> Bool in
                guard let rank1 = bpeRanks[bp1], let rank2 = bpeRanks[bp2] else {
                    return false
                }
                return rank1 < rank2
            }) else {
                break
            }
            let first = bigram.a
            let second = bigram.b
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i..<word.count].firstIndex(of: first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i..<word.count])
                    break
                }

                if word[i] == first && i < word.count - 1 && word[i+1] == second {
                    newWord.append(first+second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 {
                break
            } else {
                pairs = Array(getPairs(word: word))
            }
        }
        return word.joined(separator: " ")
    }

    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for token in self.byteEncode(text: text) {
            let xx = self.bpe(token: token).split(separator: " ").map { String($0) }
            tokens.append(contentsOf: xx)
        }
        return tokens
    }

    /// Main entry point - encodes text to token IDs
    func encode(text: String) throws -> [Int] {
        return try tokenize(text: text).map { token in
            guard let id = encoder[token] else {
                throw GPT2Error.tokenizationFailed("Unknown token: \(token)")
            }
            return id
        }
    }

    /// Decode token IDs back to text
    func decode(tokens: [Int]) throws -> String {
        let textParts = try tokens.map { tokenId -> String in
            guard let token = decoder[tokenId] else {
                throw GPT2Error.invalidToken(tokenId)
            }
            return token
        }
        let text = textParts.joined(separator: "")

        let utfCodepoints = try text.map { char -> UInt8 in
            guard let codepoint = byteDecoder[String(char)] else {
                throw GPT2Error.tokenizationFailed("Unknown character in decoded text: \(char)")
            }
            return codepoint
        }
        return String(decoding: utfCodepoints, as: UTF8.self)
    }
}
