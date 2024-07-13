import spacy
from spacy import displacy
from IPython.display import display, HTML
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Process the text and print each token with its POS tag and dependency relation
doc1 = nlp("Company Y is planning to acquire stake in X company for $23 billion")
print("Tokens with POS and dependency labels:")
for token in doc1:
    print(token.text, token.pos_, token.dep_)

# Creating an nlp object
doc2 = nlp("He went to play cricket with friends in the stadium")

# Fetching the names of the pipeline components
pipe_names = nlp.pipe_names
print("Pipeline components:", pipe_names)

# Disable the 'tagger' and 'parser' components
nlp.disable_pipes('tagger', 'parser')

# Get the names of the active pipeline components after disabling
pipe_names_disabled = nlp.pipe_names
print("Active pipeline components after disabling:", pipe_names_disabled)

# Re-enable the 'tagger' and 'parser' components
nlp.enable_pipe('tagger')
nlp.enable_pipe('parser')

# Verify that the components are re-enabled
pipe_names_enabled = nlp.pipe_names
print("Active pipeline components after re-enabling:", pipe_names_enabled)

# Create an nlp object for the next text
doc3 = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion")

# Print each token in the doc
print("Tokens:")
for token in doc3:
    print(token.text)

# Iterate over the tokens and print the token and its part-of-speech tag
print("Tokens with POS tags and explanations:")
for token in doc3:
    print(token.text, token.tag_, token.pos_, spacy.explain(token.tag_))

# Iterate over the tokens and print the token and its dependency label
print("Tokens with dependency labels:")
for token in doc3:
    print(token.text, "-->", token.dep_)

# Dependency label explanations
dep_labels = {
    "nsubj": "nominal subject",
    "ROOT": "root",
    "aux": "auxiliary",
    "advcl": "adverbial clause modifier",
    "dobj": "direct object"
}

print("Dependency label explanations:")
for label, explanation in dep_labels.items():
    print(f"{label}: {explanation}")

# Create an nlp object for the final text
doc4 = nlp("The children are playing in the garden and eating apples.")

# Iterate over the tokens and print the token and its lemma
print("Tokens with lemmas:")
for token in doc4:
    print(f"{token.text} ({token.pos_}) --> {token.lemma_}")

# Render the dependency visualization in a file and open it in a browser
svg = displacy.render(doc4, style="dep")
with open("dependency_visualization.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("Dependency visualization saved as dependency_visualization.svg.")

# Process the text
doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion. This is India. India is great.")

# Split the document into sentences
sentences = list(doc.sents)

# Count and print the number of sentences
num_sentences = len(sentences)
print("Number of sentences:", num_sentences)

# Print each sentence
for sentence in sentences:
    print(sentence.text)
#Named entities in the document
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

doc = nlp(u"""The Amazon rainforest,[a] alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations.

The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela. Four nations have "Amazonas" as the name of one of their first-level administrative regions and France uses the name "Guiana Amazonian Park" for its rainforest protected area. The Amazon represents over half of the planet's remaining rainforests,[2] and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.[3]

Etymology
The name Amazon is said to arise from a war Francisco de Orellana fought with the Tapuyas and other tribes. The women of the tribe fought alongside the men, as was their custom.[4] Orellana derived the name Amazonas from the Amazons of Greek mythology, described by Herodotus and Diodorus.[4]

History
See also: History of South America § Amazon, and Amazon River § History
Tribal societies are well capable of escalation to all-out wars between tribes. Thus, in the Amazonas, there was perpetual animosity between the neighboring tribes of the Jivaro. Several tribes of the Jivaroan group, including the Shuar, practised headhunting for trophies and headshrinking.[5] The accounts of missionaries to the area in the borderlands between Brazil and Venezuela have recounted constant infighting in the Yanomami tribes. More than a third of the Yanomamo males, on average, died from warfare.[6]""")

# Extract and display entities with their labels(ctrl+c to close the server)
entities = [(ent.text, ent.label_, ent.label) for ent in doc.ents]

# Print the entities
for entity in entities:
    print(entity)
displacy.serve(doc, style="ent")

# Load the large English model with word vectors
nlp = spacy.load("en_core_web_lg")
# Process a text
tokens = nlp("dog cat banana afskfsd")
# Iterate over tokens and print their properties
for token in tokens:
    print(token.text, 
          token.has_vector,  #
          token.vector_norm,
          token.is_oov       
         )
    
# Process the text
tokens = nlp("dog cat banana")

# Compute and print similarity between each pair of tokens
for token1 in tokens:
    for token2 in tokens:
        print(f"{token1.text} - {token2.text}: {token1.similarity(token2):.4f}")

#Using Tab Seperated Values
df_amazon = pd.read_csv("C:/Users/jorda/OneDrive/Desktop/CodeTech/NATURAL LANGUAGE PROCESSING/3 SPACY/amazon_alexa.tsv", sep="\t")
# Display the first few rows of the DataFrame
print(df_amazon.head())

#Shape of the DataFrame
print("Shape of the DataFrame:", df_amazon.shape)

#Info of the DataFrame
print("\nDataFrame Information:")
df_amazon.info()

# Count of unique values in the 'feedback' column
feedback_counts = df_amazon.feedback.value_counts()

print(feedback_counts)

# List of punctuation marks
punctuations = string.punctuation

# List of stopwords
stop_words = nlp.Defaults.stop_words

# Creating tokenizer function
def spacy_tokenizer(sentence):
    # Create a spaCy document object
    doc = nlp(sentence)

    # Lemmatize each token and convert each token into lowercase
    tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc]

    # Remove stop words and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations]

    # Return preprocessed list of tokens
    return tokens

# Example 
sentence = "The sky is blue but the quick fox jumps over the wall."
print(spacy_tokenizer(sentence))

# Function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

# Custom transformer class
class Predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Example usage
if __name__ == "__main__":
    # Sample data
    data = ["  Hello There!  ", "  This is a Trial.  ", "Clean ME UP!  "]
    
    # Initialize and apply the custom transformer
    transformer = Predictors()
    cleaned_data = transformer.transform(data)
    
    print(cleaned_data)
    
#CountVectorizer with custom tokenizer
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,1))
#TfidfVectorizer with custom tokenizer
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

# Example test
documents = [
    "Reliance is looking at buying U.K. based analytics startup for $7 billion.",
    "This is India. India is great.",
    "The Amazon rainforest is a large tropical rainforest."
]

# Fit and transform the document
X = tfidf_vector.fit_transform(documents)

# Convert to DataFrame for easy visualization
df_tfidf = pd.DataFrame(X.toarray(), columns=tfidf_vector.get_feature_names_out())
print(df_tfidf)



# Features (text data) and labels (feedback) from the DataFrame
X = df_amazon['verified_reviews']  # Data to analyze
ylabels = df_amazon['feedback']    # Labels/categories for classification

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)

# Define Bag of Words vectorizer
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,1))

# Define the classifier (example with MultinomialNB)
classifier = MultinomialNB()

# Define stop words and punctuation
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Tokenizer function
def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens

# Custom transformer
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Ensure the text is a string before processing
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

# Handle missing or non-string values
df_amazon['verified_reviews'] = df_amazon['verified_reviews'].fillna("")
df_amazon['verified_reviews'] = df_amazon['verified_reviews'].astype(str)

# Features and labels
X = df_amazon['verified_reviews']
ylabels = df_amazon['feedback']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)

# Define Bag of Words vectorizer
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,1))

# Define the classifier (example with MultinomialNB)
classifier = MultinomialNB()

# Create pipeline using Bag of Words
pipe = Pipeline([
    ("cleaner", predictors()),        # Text cleaning step
    ('vectorizer', bow_vector),      # Text vectorization step
    ('classifier', classifier)       # Classification step
])

# Train the model
pipe.fit(X_train, y_train)

# Predict with the test dataset
predicted = pipe.predict(X_test)

# Model Accuracy and other metrics
print("Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Precision:", metrics.precision_score(y_test, predicted, average='weighted')) # Use average='weighted' for multi-class
print("Recall:", metrics.recall_score(y_test, predicted, average='weighted')) # Use average='weighted' for multi-class
print("F1 Score:", metrics.f1_score(y_test, predicted, average='weighted')) # F1 Score is often used alongside precision and recall