import java.util.*;
import java.util.regex.*;

public class Tokenizer {
    private Map<String, Integer> vocab;
    private Map<Integer, String> reverseVocab;
    private int unkTokenId;

    public Tokenizer(Set<String> vocabulary) {
        vocab = new HashMap<>();
        reverseVocab = new HashMap<>();
        int id = 0;

        for (String word : vocabulary) {
            vocab.put(word, id);
            reverseVocab.put(id, word);
            id++;
        }

        unkTokenId = id;
        vocab.put("<UNK>", unkTokenId);
        reverseVocab.put(unkTokenId, "<UNK>");
    }

    // Tokenize text into words with improved regex and case normalization
    private List<String> tokenize(String text) {
        text = text.toLowerCase();

        String regex = "\\b\\w+\\b|[.,!?]";
        Matcher matcher = Pattern.compile(regex).matcher(text);

        List<String> tokens = new ArrayList<>();
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

    // Convert tokens to IDs
    public List<Integer> encode(String text) {
        List<String> tokens = tokenize(text);
        List<Integer> tokenIds = new ArrayList<>();
        for (String token : tokens) {
            tokenIds.add(vocab.getOrDefault(token, unkTokenId));
        }
        return tokenIds;
    }

    // Convert IDs back to tokens
    public String decode(List<Integer> tokenIds) {
        StringBuilder decodedText = new StringBuilder();
        for (int id : tokenIds) {
            decodedText.append(reverseVocab.getOrDefault(id, "<UNK>")).append(" ");
        }
        return decodedText.toString().trim();
    }

    public static void main(String[] args) {
        // Example vocabulary
        Set<String> vocabulary = new HashSet<>(Arrays.asList("hello", "world", "this", "is", "a", "test", "."));

        // Initialize tokenizer
        Tokenizer tokenizer = new Tokenizer(vocabulary);

        // Example text
        String text = "Hello, world. This is a test!";
        List<Integer> encoded = tokenizer.encode(text);
        System.out.println("Encoded: " + encoded);

        // Decode back to text
        String decoded = tokenizer.decode(encoded);
        System.out.println("Decoded: " + decoded);
    }
}
