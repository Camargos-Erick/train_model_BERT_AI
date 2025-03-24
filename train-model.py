import numpy
import pandas
import sqlite3
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Conecta ao banco de dados SQLite e executa uma consulta para obter dados
# Connects to the SQLite database and executes a query to fetch data
try:
    conn = sqlite3.connect("dataset.db")

    query = "SELECT defeito, id, palavra_chave FROM arvore_diagnostico"
    dataframe = pandas.read_sql_query(query, conn)

except:
    # Imprime uma mensagem de erro se a conexão falhar
    # Prints an error message if the connection fails
    print("Falha no carregamento do banco de dados, verifique se o arquivo dataset.db está na pasta do script")


# Divide os dados em conjuntos de treinamento e teste
# Splits the data into training and testing sets
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(dataframe["descricao"].tolist(), dataframe["token"].tolist(),random_state=49, test_size=0.20, shuffle=True)

# Inicializa o tokenizer BERT
# Initializes the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")
train_token = tokenizer(df_x_train, truncation= True, padding= True, max_length= 128)
test_token = tokenizer(df_x_test, truncation= True, padding= True, max_length= 128)

# Cria datasets de treinamento e teste a partir dos tokens
# Creates training and testing datasets from the tokens
train_dataset = Dataset.from_dict({
    "input_ids": train_token["input_ids"],
    "attention_mask": train_token["attention_mask"],
    "labels": df_y_train
})

test_dataset = Dataset.from_dict({
    "input_ids": test_token["input_ids"],
    "attention_mask": test_token["attention_mask"],
    "labels": df_y_test
})

# Define o número de rótulos para a classificação
# Defines the number of labels for classification
num_labels = len(set(dataframe["token"]))
model = BertForSequenceClassification.from_pretrained("neuralmind/bert-large-portuguese-cased", num_labels=num_labels)

# Função para calcular métricas de avaliação
# Function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = numpy.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define os argumentos de treinamento
# Defines the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # Ajuste o número de épocas conforme necessário
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01, 
    logging_dir="./logs",
    evaluation_strategy="epoch",  # Avalia o modelo ao final de cada época
    save_strategy="epoch",  # Salva o modelo ao final de cada época
    load_best_model_at_end=True  

)

# Inicializa o Trainer com o modelo, argumentos de treinamento, datasets e função de métricas
# Initializes the Trainer with the model, training arguments, datasets, and metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Treina o modelo
# Trains the model
trainer.train()

# Salva o modelo treinado e o tokenizer
# Saves the trained model and tokenizer
model.save_pretrained("./modelo_bert_falhas")
tokenizer.save_pretrained("./modelo_bert_falhas")
