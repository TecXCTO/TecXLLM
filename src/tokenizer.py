# <--- PUT THE TOKENIZATION CODE GENERATED ABOVE HERE
import regex as re
from typing import Dict, List, Tuple

class ByteLevelBPETokenizer:
    def __init__(self):
        # 1. Initialize vocabulary with the 256 core native byte values (0-255)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        
        # GPT-4 regex splitter to prevent merges crossing punctuation, spaces, or numbers
        self.split_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _get_stats(self, ids_list: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Counts the frequency of all adjacent token pairs."""
        counts = {}
        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_ids(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Replaces instances of the target pair with the new merged index."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text_corpus: str, vocab_size: int):
        """Trains the tokenizer from scratch using byte pairs."""
        assert vocab_size >= 256, "Vocab size must be at least 256 to cover base bytes."
        num_merges = vocab_size - 256
        
        # Pre-tokenize text using the regex pattern into separate words/chunks
        text_chunks = self.split_pattern.findall(text_corpus)
        
        # Convert raw strings into initial UTF-8 integer byte arrays
        ids_list = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Iteratively merge the most frequent pairs
        for i in range(num_merges):
            stats = self._get_stats(ids_list)
            if not stats:
                break  # No more pairs left to merge
                
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            
            # Record the merge rule
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Apply merge across all document chunks
            ids_list = [self._merge_ids(ids, best_pair, new_idx) for ids in ids_list]
            print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_idx} (Bytes: {self.vocab[new_idx]})")

    def encode(self, text: str) -> List[int]:
        """Encodes raw text containing letters, numbers, and symbols into token IDs."""
        text_chunks = self.split_pattern.findall(text)
        final_ids = []
        
        for chunk in text_chunks:
            # Convert chunk to raw bytes
            chunk_ids = list(chunk.encode("utf-8"))
            
            # Continually apply known merge rules in chronological order
            while len(chunk_ids) >= 2:
                stats = zip(chunk_ids, chunk_ids[1:])
                # Find the merge pair that happened earliest during training
                pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
                
                if pair not in self.merges:
                    break # No more merge rules apply to this chunk
                    
                chunk_ids = self._merge_ids(chunk_ids, pair, self.merges[pair])
                
            final_ids.extend(chunk_ids)
            
        return final_ids

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back into an exact string representation."""
        # Join the raw byte arrays representing each token ID
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Decode bytes safely back to string, replacing corrupt fragments if any exist
        return raw_bytes.decode("utf-8", errors="replace")

# ==========================================
# TESTING THE TOKENIZER WITH ALL CHARACTER TYPES
# ==========================================
if __name__ == "__main__":
    # Create sample complex training corpus with numbers, letters, and advanced math symbols
    training_data = """
    The equation 2 + 2 = 4 is fundamental. 
    Advanced mathematics relies heavily on symbols like \u2211 (summation) and \u222b (integral).
    Calculus definitions use variables like x, y, and z across continuous real domains \u211d.
    Ancient glyphs and historical symbols include things like \u262e or complex logic rules.
    """

    print("--- Initializing and Training Tokenizer ---")
    tokenizer = ByteLevelBPETokenizer()
    
    # Train until we have a custom vocabulary size of 265 (256 base bytes + 9 learned merges)
    tokenizer.train(training_data, vocab_size=265)

    print("\n--- Vocabulary Mapping Verification ---")
    print(f"Total trained vocabulary size: {len(tokenizer.vocab)}")

    # Test sentence containing English text, numbers, and exotic symbols
    test_sentence = "Math test: 2 + 2 = 4. Let's compute \u2211 and \u222b over \u211d."
    
    print("\n--- Encoding Raw Input Text ---")
    token_ids = tokenizer.encode(test_sentence)
    print(f"Input Text: {test_sentence}")
    print(f"Generated Token IDs: {token_ids}")

    print("\n--- Decoding Token IDs Back into Text ---")
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded Output: {decoded_text}")
    
    # Verify strict equality
    assert test_sentence == decoded_text, "Error: Tokenizer lost information during parsing!"
    print("Success: Zero information lost. Completely robust.")
  
