import java.util.*;
import java.util.regex.*;

public class GPT2Tokenizer {
    private Map<String, Integer> vocab;
    private Map<Integer, String> reverseVocab;
    private int eosTokenId;

    public GPT2Tokenizer(Map<String, Integer> vocab) {
        this.vocab = vocab;
        this.reverseVocab = new HashMap<>();

        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            reverseVocab.put(entry.getValue(), entry.getKey());
        }

        // Assign <|endoftext|> token
        if (vocab.containsKey("<|endoftext|>")) {
            eosTokenId = vocab.get("<|endoftext|>");
        } else {
            throw new IllegalArgumentException("Vocabulary must contain <|endoftext|> token.");
        }
    }

    // Tokenize input using regex and BPE logic
    private List<String> tokenize(String text) {
        text = text.toLowerCase();

        String regex = "\\b\\w+\\b|[.,!?;'\"]|\\s+";
        Matcher matcher = Pattern.compile(regex).matcher(text);

        List<String> tokens = new ArrayList<>();
        while (matcher.find()) {
            String token = matcher.group().trim();
            if (!token.isEmpty()) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    // Apply byte pair encoding (placeholder logic)
    private List<String> applyBPE(String word) {
        // This placeholder splits unknown words into characters (real BPE merges frequent subwords)
        List<String> subwords = new ArrayList<>();
        for (char c : word.toCharArray()) {
            subwords.add(String.valueOf(c));
        }
        return subwords;
    }

    // Encode text into token IDs
    public List<Integer> encode(String text) {
        List<String> tokens = tokenize(text);
        List<Integer> tokenIds = new ArrayList<>();

        for (String token : tokens) {
            if (vocab.containsKey(token)) {
                tokenIds.add(vocab.get(token));
            } else {
                // Apply BPE to handle unknown words
                List<String> subwords = applyBPE(token);
                for (String subword : subwords) {
                    tokenIds.add(vocab.getOrDefault(subword, eosTokenId)); // Use <|endoftext|> as fallback
                }
            }
        }

        tokenIds.add(eosTokenId);
        return tokenIds;
    }

    // Decode token IDs back to text
    public String decode(List<Integer> tokenIds) {
        StringBuilder decodedText = new StringBuilder();

        for (int id : tokenIds) {
            if (id == eosTokenId) {
                decodedText.append("<|endoftext|>");
            } else {
                String token = reverseVocab.getOrDefault(id, "");
                decodedText.append(token);
            }
        }

        return decodedText.toString().trim();
    }

    public static void main(String[] args) {
        // Example GPT-2 vocabulary (simplified for demonstration)
        Map<String, Integer> vocabulary = new HashMap<>();
        vocabulary.put("hello", 0);
        vocabulary.put(",", 1);
        vocabulary.put("world", 2);
        vocabulary.put(".", 3);
        vocabulary.put("<|endoftext|>", 4);

        // Initialize tokenizer
        GPT2Tokenizer tokenizer = new GPT2Tokenizer(vocabulary);

        // Encode text
        String text = "Hello, world. This is a test !";
        List<Integer> encoded = tokenizer.encode(text);
        System.out.println("Encoded: " + encoded);

        // Decode back to text
        String decoded = tokenizer.decode(encoded);
        System.out.println("Decoded: " + decoded);
    }
}
