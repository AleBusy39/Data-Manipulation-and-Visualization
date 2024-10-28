# ###########################################################################################################
# Titolo della Presentazione: Analisi delle Vendite e Classificazione della Qualità delle Mele
# ###########################################################################################################
#
# Introduzione:
# Questo script esegue un'analisi delle vendite suddivisa per città, genere e categoria di prodotto, insieme a
# una classificazione della qualità delle mele utilizzando un modello di machine learning.
#
# Obiettivi:
# - Analizzare i dati di vendita per identificare tendenze e differenze tra generi e categorie di prodotto.
# - Applicare il modello Random Forest per prevedere la qualità delle mele basandosi sulle caratteristiche.
#
# Metodologia:
# 1. Analisi delle vendite per genere e città.
# 2. Creazione di grafici a barre per visualizzare le vendite.
# 3. Classificazione delle mele e valutazione della performance del modello tramite matrice di confusione.
# 4. Visualizzazione del trend delle vendite mensili nel tempo.

# Import delle librerie necessarie per l'analisi e il machine learning
from sklearn.model_selection import train_test_split                                    # Funzione per suddividere il dataset in training set e test set
from sklearn.ensemble import RandomForestClassifier                                     # Classificatore Random Forest per problemi di classificazione
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix # Funzioni per valutare le performance del modello di classificazione
from sklearn.linear_model import LinearRegression                                       # Modello di regressione lineare per problemi di regressione
import pandas as pd                                                                     # Pandas è utilizzato per la manipolazione e l'analisi dei dati
import matplotlib.pyplot as plt                                                         # Matplotlib per la creazione di grafici
import seaborn as sns                                                                   # Seaborn per la visualizzazione avanzata dei dati
from matplotlib.ticker import FuncFormatter                                             # FuncFormatter per formattare gli assi dei grafici

def thousand_separator_for_plot(x, pos):                                                # Funzione per formattare i numeri per gli assi dei grafici
    """
    Formatta un numero per l'asse di un grafico con il punto come separatore delle migliaia.

    Args:
        x: Il numero da formattare.
        pos (int): La posizione dell'etichetta sull'asse (richiesta da matplotlib, ma non usata direttamente).

    Returns:
        str: Il numero formattato con il punto come separatore delle migliaia.
    """
    return f'{int(x):,}'.replace(',', '.')                                              # Formatta il numero con il punto come separatore delle migliaia per i grafici.

def thousand_separator(x):                                                              # Funzione per formattare i numeri per l'output
    """
    Formatta un numero con il punto come separatore delle migliaia per l'output.

    Args:
        x: Il numero da formattare.

    Returns:
        str: Il numero formattato con il punto come separatore delle migliaia.
    """
    return f'{int(x):,}'.replace(',', '.')                                              # Formatta il numero con il punto come separatore delle migliaia per l'output.

def etichetta():                                                                        # Funzione per formattare l'etichetta dei grafici a barre
    """
    Aggiunge etichette su ciascuna barra di un grafico a barre.
    Itera su tutte le barre del grafico, aggiungendo un'etichetta sopra ogni barra,
    con il valore della larghezza della barra formattato con il punto come separatore delle migliaia.

    Modifica lo stile del testo per l'etichetta, posizionandola al centro della barra orizzontalmente e verticalmente.

    Args:
        None

    Returns:
        None
    """
    for p in ax.patches:
        ax.annotate(f'{p.get_width():,.0f}'.replace(',', '.'),                          # Utilizza 'annotate' per aggiungere un'etichetta sopra ogni barra.
                                      (p.get_width(), p.get_y() + p.get_height() / 2),  # Posizione dell'etichetta sulla barra, al centro verticale.
                                      ha='center', va='center',                         # Allineamento orizzontale e verticale al centro.
                                      color='black', fontsize=10, fontweight='bold')    # Stile del testo dell'etichetta.
   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Caricamento File.csv
sales_data = pd.read_csv(r'C:\Users\alessandro\Desktop\file famiglia\Alessandro\Corso Start2Impact\08 - Advanced Analytics\supermarket_sales - Copia.csv')  #Caricamento primo file csv
apple_data = pd.read_csv(r'C:\Users\alessandro\Desktop\file famiglia\Alessandro\Corso Start2Impact\08 - Advanced Analytics\apple_quality.csv')              #Caricamento secondo file csv

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

city_customer_sales = sales_data.groupby(['City', 'Customer type'])['Total'].sum().reset_index() # Raggruppa il DataFrame sales_data in base alle colonne 'City' e 'Customer type', calcola la somma totale delle vendite nella colonna 'Total e poi resetta l'indice per ottenere un DataFrame ben formattato.
customer_sales_total = sales_data.groupby('Customer type')['Total'].sum().reset_index()          # Raggruppa il DataFrame sales_data in base alla colonna 'Customer type', calcola la somma totale delle vendite nella colonna 'Total e poi resetta l'indice per ottenere un DataFrame ben formattato.
city_total_sales = sales_data.groupby('City')['Total'].sum().reset_index()                       # Raggruppa il DataFrame sales_data in base alla colonna 'City', calcola la somma totale delle vendite nella colonna 'Total e poi resetta l'indice per ottenere un DataFrame ben formattato.

# Stampa delle vendite per città e tipo di cliente
print("Vendite per città e tipo di cliente:")
for index, row in city_customer_sales.iterrows():                                                   # Itera su ogni riga del DataFrame 'city_customer_sales' utilizzando iterrows().
    print(f"{row['City']} ({row['Customer type']}): {thousand_separator(round(row['Total'], 0))}")  # Stampa il nome della città, il tipo di cliente (Member o Normal)e il totale delle vendite formattato con separatore delle migliaia.

# Visualizzazione delle vendite per città e tipo di cliente
plt.figure(figsize=(10, 6))                                                                             # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
ax = sns.barplot(data=city_customer_sales, x='Total', y='City', hue='Customer type', palette='muted')   # Crea un grafico a barre utilizzando Seaborn, con le vendite totali per città e colorato per tipo di cliente.
plt.title('Vendite per Tipo di Cliente e Città')                                                        # Imposta il titolo del grafico.
plt.xlabel('Vendite Totali')                                                                            # Etichetta l'asse X come 'Vendite Totali'.
plt.ylabel('Città')                                                                                     # Etichetta l'asse Y come 'Città'.
etichetta()                                                                                             # Funzione richiamata per formattare l'etichetta dei grafici a barre

# Formattazione dell'asse X
ax.xaxis.set_major_formatter(FuncFormatter(thousand_separator_for_plot))                        # Questo formatter personalizza i valori sull'asse X, utilizzando la funzione 'thousand_separator_for_plot' per formattare i numeri con un separatore delle migliaia (es. 1.000, 2.000, ecc.).
plt.legend(title='Customer type', bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)    # 'plt.legend()' crea una legenda per il grafico -'title' imposta il titolo della legenda come "Customer type", 'bbox_to_anchor' specifica la posizione della legenda nel grafico,'loc' specifica la posizione relativa della legenda, impostata su 'upper left' per posizionarla nell'angolo in alto a sinistra della casella di ancoraggio e 'borderaxespad' determina la distanza tra la legenda e l'asse del grafico; impostato a 0 per attaccare la legenda direttamente all'asse.
plt.show()                                                                                      # Mostra il grafico finale

# Storytelling delle vendite per città e tipo di cliente
print("\n VENDITE PER CITTA' E TIPO DI CLIENTE \n"
      "Mandalay (Member): 162.487.983\n"
      "Mandalay (Normal): 129.422.223\n"
      "Mandalay mostra un totale di vendite di 162.487.983 per i membri, "
      "che è significativamente più alto rispetto alle vendite per i clienti normali, pari a 129.422.223. "
      "Questo suggerisce che i clienti membri contribuiscono in modo più sostanziale alle vendite totali in questa città, "
      "il che potrebbe indicare l'efficacia di programmi di fidelizzazione o offerte specifiche per i membri.\n"
      "Naypyitaw (Member): 182.687.148\n"
      "Naypyitaw (Normal): 120.989.946\n"
      "Anche a Naypyitaw, le vendite per i membri (182.687.148) superano quelle per i normali clienti (120.989.946). "
      "Il divario tra le vendite dei membri e quelle dei normali clienti è evidente e potrebbe suggerire "
      "che le strategie di marketing e vendita siano particolarmente efficaci per attrarre membri in questa città.\n"
      "Yangon (Member): 121.135.140\n"
      "Yangon (Normal): 160.343.925\n"
      "A Yangon, la situazione è diversa. Le vendite per i clienti normali (160.343.925) superano quelle per i membri (121.135.140). "
      "Questo potrebbe indicare una base di clienti normali più forte o un interesse minore "
      "nei programmi di membership.\n\n")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Stampa delle vendite totali per tipo di cliente
print("\nVendite totali per tipo di cliente:")
for index, row in customer_sales_total.iterrows():                                       # Itera su ogni riga del DataFrame 'customer_sales_total' utilizzando iterrows().
    print(f"{row['Customer type']}: {thousand_separator(round(row['Total'], 0))}")       # Stampa il tipo di cliente (Member o Normal) e il totale delle vendite formattato con separatore delle migliaia.

# Visualizzazione delle vendite totali per tipo di cliente
plt.figure(figsize=(8, 5))                                      # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
# Crea un grafico a barre utilizzando Seaborn
ax = sns.barplot(data=customer_sales_total, 
                                       x='Total',               # 'data' è il DataFrame che contiene i dati da visualizzare.
                                       y='Customer type',       # 'y' rappresenta la colonna 'Customer type' per la tipologia di cliente sull'asse Y.
                                       hue='Customer type',     # Imposta hue per la legenda.
                                       palette='muted',         # 'palette' specifica il tema di colori da usare per le barre.
                                       legend=False)            # Disabilita la legenda predefinita.

plt.title('Vendite Totali per Tipo di Cliente')                 # Imposta il titolo del grafico.
plt.xlabel('Vendite Totali')                                    # Etichetta l'asse X come 'Vendite Totali'.
plt.ylabel('Tipo di Cliente')                                   # Etichetta l'asse Y come 'Tipo di Cliente'.
etichetta()                                                     # Funzione richiamata per formattare l'etichetta dei grafici a barre
ax.xaxis.set_major_formatter(FuncFormatter(thousand_separator_for_plot))  # Personalizza i valori sull'asse X con il separatore delle migliaia.
plt.show()                                                      # Mostra il grafico.

# Storytelling delle vendite per tipo di cliente
print("\nVENDITE TOTALI PER TIPO DI CLIENTE \n"
      "Member: 466.310.271\n"
      "Normal: 410.756.094\n"
      "Le vendite totali per tipo di cliente mostrano che i membri contribuiscono significativamente di più alle vendite complessive, "
      "con un totale di 466.310.271 contro 410.756.094 dei clienti normali. Questo è un dato "
      "positivo, in quanto suggerisce che l'implementazione di strategie di fidelizzazione ha portato a vendite maggiori. "
      "Tuttavia, è importante anche considerare la proporzione di clienti normali e trovare modi per "
      "convertirli in membri.\n")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Stampa delle vendite totali per città
print("\nTotale vendite per città:")
for index, row in city_total_sales.iterrows():                             # Itera su ogni riga del DataFrame 'city_total_sales' utilizzando iterrows().
    print(f"{row['City']}: {thousand_separator(round(row['Total'], 0))}")  # Stampa il nome della città e il totale delle vendite formattato con separatore delle migliaia.

plt.figure(figsize=(10, 6))                             # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
# Crea un grafico a barre utilizzando Seaborn
ax = sns.barplot(data=city_total_sales,                 # 'data' è il DataFrame che contiene i dati da visualizzare.
                                    x='Total',          # 'x' rappresenta la colonna 'Total' per le vendite totali sull'asse X.
                                    y='City',           # 'y' rappresenta la colonna 'City' per le città sull'asse Y.
                                    hue='City',         # Imposta hue per la legenda.
                                    palette='muted',    # 'palette' specifica il tema di colori da usare per le barre. 
                                    legend=False)       # Disabilita la legenda predefinita.

plt.title('Vendite Totali per Città')   # Imposta il titolo del grafico. Questo appare in cima al grafico e fornisce un contesto al lettore.
plt.xlabel('Vendite Totali')            # Etichetta l'asse X con la descrizione 'Vendite Totali'. Questo aiuta a capire cosa rappresentano i valori sull'asse orizzontale.
plt.ylabel('Città')                     # Etichetta l'asse Y con la descrizione 'Città'. Questo aiuta a capire quale categoria è rappresentata sull'asse verticale.
etichetta()                             # Funzione richiamata per formattare l'etichetta dei grafici a barre
ax.xaxis.set_major_formatter(FuncFormatter(thousand_separator_for_plot))  # Formatta l'asse X per mostrare i valori con un separatore delle migliaia
plt.show()                              # Visualizza il grafico

# Storytelling delle vendite per città
print("TOTALE VENDITE PER CITTA' \n"
      "Mandalay: 291.910.206\n"
      "Naypyitaw: 303.677.094\n"
      "Yangon: 281.479.065\n"
      "In termini di vendite totali per città, Naypyitaw ha il valore più alto con 303.677.094, seguita da Mandalay (291.910.206) "
      "e Yangon (281.479.065). Questo potrebbe indicare che Naypyitaw sta performando meglio nel "
      "complesso, il che potrebbe essere dovuto a fattori come una migliore strategia di marketing, "
      "una maggiore popolazione di clienti o una combinazione di fattori favorevoli.\n")

print("-" * 40)                                # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Analisi delle vendite per categoria di prodotto
product_sales = sales_data.groupby('Product line')['Total'].sum().reset_index().sort_values(by='Total', ascending=False) # Raggruppa i dati per la categoria di prodotto ('Product line') e somma le vendite totali ('Total') per ogni categoria, 'reset_index()' riporta il risultato in un DataFrame e 'sort_values(by='Total', ascending=False)' ordina le categorie in ordine decrescente di vendite.
print("\nVendite per categoria di prodotto:")                                                                            # Stampa un'intestazione per le vendite per categoria di prodotto.
for index, row in product_sales.iterrows():                                                                              # Itera su ogni riga del DataFrame 'product_sales'.                                                                           
    print(f"{row['Product line']}: {thousand_separator(round(row['Total'], 0))}")                                        # Per ogni categoria di prodotto, stampa il nome della categoria e il totale delle vendite formattato, 'thousand_separator(row['Total'])' applica la formattazione numerica definita per separare le migliaia e arrotonda al numero intero più vicino.

# Visualizza le vendite per categoria di prodotto tramite grafico
plt.figure(figsize=(10, 6))                                                                                                           # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
ax = sns.barplot(data=product_sales, x='Total', y='Product line', palette='pastel', hue='Product line', dodge=False, legend=False)    # Crea un grafico a barre con le vendite totali per ciascuna categoria di prodotto - 'data=product_sales' specifica il DataFrame da utilizzare. 'x' è impostato su 'Total' per le vendite totali e 'y' su 'Product line' per le categorie di prodotto, 'palette' specifica i colori per le barre e 'hue' permette di colorare le barre in base alla categoria,  'dodge=False' indica che le barre non devono essere divise e 'legend=False' disabilita la legenda.
plt.title('Vendite Totali per Categoria di Prodotto')                                                                                 # Imposta il titolo del grafico.
plt.xlabel('Vendite Totali')                                                                                                          # Etichetta l'asse X come 'Vendite Totali'.
plt.ylabel('Categoria di Prodotto')                                                                                                   # Etichetta l'asse Y come 'Categoria di Prodotto'.
etichetta()                                                                                                                           # Funzione richiamata per formattare l'etichetta dei grafici a barre
# Formattazione dell'asse X
ax.xaxis.set_major_formatter(FuncFormatter(thousand_separator_for_plot))    # Utilizza il formatter personalizzato per l'asse X, che applica il formato numerico definito in 'thousand_separator_for_plot'
plt.show()                        # Mostra il grafico finale.

# Storytelling delle vendite per categoria di prodotto
print("""
ANALISI DEI DATI

Dominanza della categoria "Health and Beauty":
La categoria Health and Beauty registra il fatturato più alto con 165.829.230. Questo potrebbe riflettere un crescente interesse e domanda per prodotti legati alla salute e alla bellezza, un trend amplificato 
anche dall'attenzione verso il benessere personale. È importante notare che i prodotti per la salute e la bellezza spesso hanno margini di profitto più elevati, il che potrebbe contribuire alla loro elevata 
performance.

Solidità degli "Electronic Accessories":
Gli Electronic Accessories seguono da vicino con 153.447.336. Questa categoria è in continua espansione grazie alla crescente digitalizzazione e all'aumento dell'uso di dispositivi tecnologici. La domanda di 
accessori per smartphone, computer e altri dispositivi elettronici è costante. La popolarità di gadget e accessori tecnologici suggerisce anche che ci siano opportunità per cross-selling e upselling con prodotti 
complementari.

Interesse per "Sports and Travel":
Con 152.052.516, la categoria Sports and Travel è anch'essa significativa. L'aumento dell'interesse per attività all'aperto e sport, soprattutto post-pandemia, può contribuire a questa cifra. Potrebbe essere utile 
promuovere prodotti legati a stili di vita attivi e viaggi, magari in sinergia con il marketing di esperienze e avventure.

"Fashion Accessories" e "Home and Lifestyle":
Le vendite di Fashion Accessories e Home and Lifestyle sono rispettivamente 146.580.231 e 137.036.466. Queste categorie riflettono tendenze culturali e stili di vita. I clienti sono sempre più interessati a prodotti 
che migliorano la loro qualità della vita e il loro stile. Promozioni strategiche o collaborazioni con influencer possono incrementare l'interesse e le vendite in queste categorie.

Crescita di "Food and Beverages":
Infine, la categoria Food and Beverages ha registrato vendite di 122.120.586. Anche se è la categoria con le vendite più basse rispetto alle altre, essa può rappresentare un mercato in crescita, soprattutto se 
si considera l'interesse crescente per cibi sani e bevande artigianali. Si potrebbe considerare di diversificare l'offerta in questa categoria, introducendo nuovi prodotti o varianti per attrarre una clientela 
più ampia.
""")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Raggruppa le vendite per categoria di prodotto e genere
category_gender_sales = sales_data.groupby(['Product line', 'Gender'])['Total'].sum().reset_index()     # Utilizza la funzione 'groupby' per raggruppare il DataFrame 'sales_data' per 'Product line' e 'Gender' e calcola la somma delle vendite totali ('Total') per ciascun gruppo e il risultato viene ripristinato come un nuovo DataFrame con l'indice reimpostato.

# Stampa delle vendite per categoria e genere
print("\nVendite per categoria di prodotto e genere:")                                                  # Stampa un'intestazione per le vendite per categoria e genere.
for index, row in category_gender_sales.iterrows():                                                     # Itera su ogni riga del DataFrame 'category_gender_sales' e stampiamo i risultati.
    print(f"{row['Product line']} ({row['Gender']}): {thousand_separator(round(row['Total'], 0))}")     # Per ogni riga, stampa la categoria di prodotto e il genere con il totale delle vendite formattato utilizzando la funzione 'thousand_separator' per formattare i numeri.

# Visualizzazione delle vendite per categoria e genere
plt.figure(figsize=(12, 8))                                                                             # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
ax = sns.barplot(data=category_gender_sales, x='Total', y='Product line', hue='Gender', palette='muted')# Utilizza 'seaborn' per creare un grafico a barre, dove le vendite totali sono sull'asse X, le categorie di prodotto sull'asse Y, e il colore delle barre rappresenta il genere.
plt.title('Vendite per Categoria di Prodotto e Genere')                                                 # Imposta il titolo del grafico.
plt.xlabel('Vendite Totali')                                                                            # Etichetta l'asse X come 'Vendite Totali'.
plt.ylabel('Categoria di Prodotto')                                                                     # Etichetta l'asse Y come 'Categoria di prodotto'.
etichetta()                                                                                             # Funzione richiamata per formattare l'etichetta dei grafici a barre

# Formattazione dell'asse X
ax.xaxis.set_major_formatter(FuncFormatter(thousand_separator_for_plot))  # Utilizza il formatter personalizzato per l'asse X, che applica il formato numerico definito in 'thousand_separator_for_plot'
plt.legend(title='Genere')                                                # 'plt.legend()' crea una legenda per il grafico -'title' imposta il titolo della legenda come "Genere".
plt.show()                                                                # Mostra il grafico finale

# Storytelling delle vendite per categoria di prodotto e genere
print("""

VENDITE PER CATEGORIA DI PRODOTTO E GENERE

Electronic Accessories:
Le vendite di accessori elettronici sono relativamente equilibrate tra i due generi, con una leggera preferenza per gli acquisti da parte degli uomini. In totale, le vendite sono:
- Female: 75.291.447
- Male: 78.155.889

Fashion Accessories:
La categoria degli accessori di moda mostra una forte predominanza delle donne come acquirenti, suggerendo che le donne sono più propense a spendere in questa categoria rispetto agli uomini. Le vendite sono:
- Female: 93.240.966
- Male: 53.339.265

Food and Beverages:
Anche qui, le donne mostrano una maggiore propensione a spendere in questa categoria, il che potrebbe riflettere le responsabilità tradizionali di acquisto di cibo e bevande da parte delle donne in molte culture. 
Le vendite sono:
- Female: 75.420.639
- Male: 46.699.947

Health and Beauty:
Le vendite nella categoria salute e bellezza sono più elevate tra gli uomini, il che potrebbe indicare un crescente interesse maschile verso i prodotti di bellezza e benessere, in linea con le tendenze di 
mercato recenti. Le vendite sono:
- Female: 79.495.626
- Male: 86.333.604

Home and Lifestyle:
Le donne sembrano dominare anche in questa categoria, suggerendo un forte interesse per l'arredamento e il miglioramento della casa. Le vendite sono:
- Female: 82.081.902
- Male: 54.954.564

Sports and Travel:
Qui, gli uomini mostrano una spesa superiore, il che può riflettere un maggiore coinvolgimento maschile in attività sportive e viaggi. Le vendite sono:
- Female: 68.898.060
- Male: 83.154.456

""")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Impatto del genere sulle vendite
gender_sales = sales_data.groupby('Gender')['Total'].sum().reset_index().sort_values(by='Total', ascending=False) # Raggruppa i dati per genere ('Gender') e somma le vendite totali ('Total') per ogni genere, 'reset_index()' riporta il risultato in un DataFrame e 'sort_values(by='Total', ascending=False)' ordina i generi in ordine decrescente di vendite.
print("\nVendite per genere:")                                                                                    # Stampa un'intestazione per le vendite per genere.
for index, row in gender_sales.iterrows():                                                                        # Itera su ogni riga del DataFrame 'gender_sales'.
    print(f"{row['Gender']}: {thousand_separator(round(row['Total'], 0))}")                                       # Per ogni genere, stampa il nome del genere e il totale delle vendite formattato, 'thousand_separator(row['Total'])' applica la formattazione numerica definita per separare le migliaia e arrotonda al numero intero più vicino.

# Visualizzazione vendite per genere
plt.figure(figsize=(10, 6))                                                                                           # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
ax = sns.barplot(data=gender_sales, x='Total', y='Gender', palette='pastel', hue='Gender', dodge=False, legend=False) # Crea un grafico a barre con le vendite totali per ciascun genere - 'data=gender_sales' specifica il DataFrame da utilizzare, 'x' è impostato su 'Total' per le vendite totali e 'y' su 'Gender' per i generi, 'palette' specifica i colori per le barre e 'hue' permette di colorare le barre in base al genere, 'dodge=False' indica che le barre non devono essere divise e 'legend=False' disabilita la legenda.
plt.title('Vendite Totali per Genere')                                                                                # Imposta il titolo del grafico.
plt.xlabel('Vendite Totali')                                                                                          # Etichetta l'asse X come 'Vendite Totali'.
plt.ylabel('Genere')                                                                                                  # Etichetta l'asse Y come 'Genere'.
etichetta()                                                                                                           # Funzione richiamata per formattare l'etichetta dei grafici a barre

# Formattazione dell'asse X
ax.xaxis.set_major_formatter(FuncFormatter(thousand_separator_for_plot))       # Utilizza il formatter personalizzato per l'asse X, che applica il formato numerico definito in 'thousand_separator_for_plot'
plt.show()                                                                     # Mostra il grafico finale.

# Storytelling dell' impatto del genere sulle vendite
print("""

VENDITE TOTALI PER GENERE

Nel complesso, le vendite totali indicano che le donne hanno speso significativamente di più rispetto agli uomini, con un totale di:
- Female: 474.428.640
- Male: 402.637.725

Questo dato può suggerire che, sebbene gli uomini mostrino una spesa significativa in alcune categorie (come salute e bellezza, e sport e viaggi), le donne hanno una spesa più elevata in generale, 
in particolare nei settori della moda, cibo e articoli per la casa.

Considerazioni Finali

Preferenze di Acquisto: I dati suggeriscono che le donne tendono a investire di più in moda, cibo e lifestyle, mentre gli uomini sembrano più interessati ai prodotti di bellezza e agli sport. 
Ciò riflette le tendenze sociali e culturali attuali.
Crescita nel Settore Maschile: La crescita delle vendite in categorie come salute e bellezza per gli uomini potrebbe rappresentare una tendenza emergente che potrebbe essere ulteriormente esplorata 
per potenziali opportunità di mercato.

""")

print("-" * 40)                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Raggruppa le vendite per città, genere e categoria di prodotto
city_gender_category_sales = sales_data.groupby(['City', 'Gender', 'Product line'])['Total'].sum().reset_index()# Utilizza la funzione 'groupby' per raggruppare il DataFrame 'sales_data' in base a 'City', 'Gender' e 'Product line'. calcola la somma delle vendite totali ('Total') per ciascun gruppo e resettiamo l'indice per ottenere un nuovo DataFrame.
print("Vendite per città, genere e categoria di prodotto:")                                                     # Stampa un'intestazione per le vendite per città, genere e categoria di prodotto.

for index, row in city_gender_category_sales.iterrows():                                                        # Itera su ogni riga del DataFrame 'city_gender_category_sales'.
    print(f"{row['City']} ({row['Gender']}, {row['Product line']}): {int(row['Total']):,.0f}".replace(',', '.'))# Stampa il nome della città, il genere, la categoria di prodotto e il totale delle vendite formattato con il separatore delle migliaia (.) in formato leggibile.
  
# Crea un grafico separato per ogni categoria di prodotto
product_lines = city_gender_category_sales['Product line'].unique()                                             # Estrea le categorie di prodotto uniche dal DataFrame 'city_gender_category_sales' per poterle iterare.
                                        
for product in product_lines:                                                                                       
    product_data = city_gender_category_sales[city_gender_category_sales['Product line'] == product]            # Filtra i dati per la categoria di prodotto corrente.
    plt.figure(figsize=(12, 8))                                                                                 # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
    ax = sns.barplot(data=product_data, x='Total', y='City', hue='Gender', palette='pastel', dodge=True)        # Crea un grafico a barre utilizzando Seaborn, dove le vendite totali sono sull'asse X, le città sono sull'asse Y e il colore delle barre rappresenta il genere.
    plt.title(f'Vendite per Città e Genere - Categoria: {product}')                                             # Imposta il titolo del grafico.
    plt.xlabel('Vendite Totali')                                                                                # Etichetta l'asse X come 'Vendite Totali'.
    plt.ylabel('Città')                                                                                         # Etichetta l'asse Y come 'Città'.
    etichetta()                                                                                                 # Funzione richiamata per formattare l'etichetta dei grafici a barre                                                                                                                                                                                                                                                                 
    # Formattazione dell'asse X
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,.0f}'.replace(',', '.')))                # Utilizza il formatter personalizzato per l'asse X.
    plt.legend(title='Genere')                                                                                  # 'plt.legend()' crea una legenda per il grafico -'title' imposta il titolo della legenda come "Genere".
    plt.show()                                                                                                  # Mostra il grafico finale
    
# Storytelling delle vendite per città, genere e categoria di prodotto
print("""

VENDITE PER CITTA', GENERE E CATEGORIA DI PRODOTTO

1. Mandalay
La città di Mandalay presenta un totale di vendite diversificate e un buon equilibrio tra generi.

    Analisi per Categoria:
    
    - **Electronic Accessories**:
        Femminile: 32.944.695
        Maschile: 18.398.436
        **Osservazione**: Le donne tendono a spendere di più in elettronica, indicando una crescente presenza femminile nel mercato tecnologico.
        
    - **Fashion Accessories**:
        Femminile: 25.825.569
        Maschile: 26.589.885
        **Osservazione**: Gli uomini superano le donne in questa categoria, suggerendo un potenziale interesse per la moda maschile che potrebbe essere ulteriormente esplorato.
        
    - **Health and Beauty**:
        Femminile: 25.554.081
        Maschile: 41.260.023
        **Osservazione**: Gli uomini hanno speso significativamente di più in questa categoria, evidenziando un cambiamento nei comportamenti di acquisto maschili che pongono maggiore attenzione alla cura personale.
        
    - **Home and Lifestyle**:
        Femminile: 28.972.818
        Maschile: 11.815.692
        **Osservazione**: Qui le donne dominano le vendite, suggerendo che le decisioni di acquisto relative alla casa sono spesso influenzate dal genere.
        
    - **Sports and Travel**:
        Femminile: 21.800.310
        Maschile: 22.841.049
        **Osservazione**: Le vendite sono abbastanza equilibrate, ma gli uomini leggermente in vantaggio potrebbero indicare un interesse maggiore per le attività all'aperto.

2. Naypyitaw
Naypyitaw mostra una diversificazione simile, ma con alcune categorie che si distinguono per vendite elevate.

    Analisi per Categoria:
    
    - **Electronic Accessories**:
        Femminile: 26.854.191
        Maschile: 34.611.339
        **Osservazione**: Gli uomini qui sono nettamente in testa, suggerendo una maggiore attrazione verso la tecnologia rispetto a Mandalay.
        
    - **Fashion Accessories**:
        Femminile: 38.939.145
        Maschile: 20.408.829
        **Osservazione**: Le donne dominano in modo significativo, il che indica che Naypyitaw potrebbe essere un buon mercato per le aziende che si concentrano su prodotti femminili.
        
    - **Food and Beverages**:
        Femminile: 34.289.661
        Maschile: 10.883.523
        **Osservazione**: Le donne mostrano un forte interesse per il cibo e le bevande, il che potrebbe essere un indicativo di preferenze più marcate nella spesa alimentare.
       
    - **Health and Beauty**:
        Femminile: 25.557.420
        Maschile: 25.760.007
        **Osservazione**: Qui le vendite sono quasi equivalenti, suggerendo un interesse simile tra i generi per i prodotti di bellezza.
        
    - **Home and Lifestyle**:
        Femminile: 18.023.985
        Maschile: 18.067.917
        **Osservazione**: Quasi parità, il che suggerisce che entrambe le fasce di genere investono in prodotti per la casa.
        
    - **Sports and Travel**:
        Femminile: 31.301.046
        Maschile: 18.980.031
        **Osservazione**: Le donne mostrano un interesse marcato, probabilmente riflettendo l'aumento dell'attività fisica e dell'interesse per il benessere.

3. Yangon
Yangon presenta vendite più basse in alcune categorie rispetto a Mandalay e Naypyitaw.

    Analisi per Categoria:
        
    - **Electronic Accessories**:
        Femminile: 15.492.561
        Maschile: 25.146.114
        **Osservazione**: Gli uomini hanno speso di più, ma le vendite totali sono inferiori rispetto a Mandalay e Naypyitaw, suggerendo un mercato della tecnologia più limitato.
        
    - **Fashion Accessories**:
        Femminile: 28.476.252
        Maschile: 6.340.551
        **Osservazione**: Le donne dominano in modo schiacciante, suggerendo che Yangon potrebbe avere opportunità per le aziende di moda femminile.
        
    - **Food and Beverages**:
        Femminile: 18.779.943
        Maschile: 22.259.811
        **Osservazione**: Gli uomini hanno speso di più, indicando un potenziale cambiamento nei modelli alimentari.
        
    - **Health and Beauty**:
        Femminile: 28.384.125
        Maschile: 19.313.574
        **Osservazione**: Le donne mostrano un forte interesse, indicando un potenziale mercato per i prodotti di bellezza.
        
    - **Home and Lifestyle**:
        Femminile: 35.085.099
        Maschile: 25.070.955
        **Osservazione**: Le vendite per le donne in questa categoria sono superiori, suggerendo che le donne a Yangon investono significativamente in questo segmento.
        
    - **Sports and Travel**:
        Femminile: 15.796.704
        Maschile: 41.333.376
        **Osservazione**: Gli uomini mostrano un forte interesse per questa categoria, il che potrebbe riflettere una cultura sportiva o ricreativa prevalente.

CONCLUSIONI E RACCOMANDAZIONI

1. **Differenze di Genere e Città**
    - **Mandalay**: Gli uomini tendono a investire di più in salute e bellezza, mentre le donne dominano in elettronica. Si suggerisce di creare campagne di marketing specifiche per attrarre entrambi 
        i generi nelle loro aree di interesse.
    - **Naypyitaw**: La predominanza femminile in fashion e food presenta opportunità per l'espansione delle linee di prodotto. Le aziende dovrebbero considerare collaborazioni con influencer locali per 
        promuovere i loro prodotti.
    - **Yangon**: Le donne sono i principali consumatori in home and lifestyle e fashion, quindi un potenziamento dell'e-commerce e delle promozioni sui social media potrebbe rivelarsi fruttuoso.

2. **Tendenze di Consumo**
    - C'è un chiaro spostamento verso la cura personale tra gli uomini, con vendite significative in Health and Beauty, suggerendo opportunità per prodotti mirati.
    - Le donne mostrano interesse per prodotti tecnologici, il che potrebbe indicare una crescente partecipazione nel mercato tecnologico.

3. **Strategie di Marketing**
    - Le campagne pubblicitarie dovrebbero essere personalizzate per ciascuna città, tenendo conto delle preferenze di genere e delle categorie di prodotto.
    - Considerare eventi o pop-up store per attirare i consumatori in modo diretto, soprattutto in città come Naypyitaw e Yangon.

4. **Innovazione dei Prodotti**
    - A seconda delle preferenze di acquisto, le aziende potrebbero voler introdurre prodotti innovativi che combinano diverse categorie per attrarre un pubblico più ampio.

5. **Monitoraggio e Adattamento**
    - Monitorare costantemente le tendenze di acquisto e adattare le strategie di prodotto e marketing in base ai dati emergenti per rimanere competitivi nel mercato.

""")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Operazioni sulle mele (e gestire i valori non numerici)
apple_data['Acidity'] = pd.to_numeric(apple_data['Acidity'], errors='coerce')               # Converte la colonna 'Acidity' in valori numerici; se ci sono errori, sostituiamo con NaN.

apple_data = apple_data.dropna()                                                            # Questo elimina tutte le righe del DataFrame che contengono valori NaN in qualsiasi colonna.

X = apple_data.drop('Quality', axis=1)                                                      # Crea il DataFrame X con tutte le colonne tranne 'Quality', che sono le nostre caratteristiche predittive.
y = apple_data['Quality']                                                                   # Crea il vettore y che contiene solo la colonna 'Quality', che è la nostra variabile target da prevedere.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # Utilizza 'train_test_split' per dividere X e y in un training set e un test set, 'test_size=0.3' significa che il 30% dei dati sarà usato per il test e il 70% per l'addestramento, 'random_state=42' assicura che la divisione sia riproducibile.

rf = RandomForestClassifier(n_estimators=100, random_state=42)                              # Crea un'istanza del modello RandomForestClassifier con 100 alberi e un seme random.
rf.fit(X_train, y_train)                                                                    # Allena il modello utilizzando il training set.

y_pred = rf.predict(X_test)                                                                 # Utilizza il modello addestrato per fare previsioni sui dati del test set.

print("\nClassificazione della qualità delle mele:")                                        # Visualizza le metriche di classificazione per confrontare le previsioni del modello con i valori reali.
print(classification_report(y_test, y_pred))                                                # Stampa precisione, richiamo e punteggio F1 per ciascuna classe.

# Visualizzazione della matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)                                                                        # Crea la matrice di confusione, 'confusion_matrix' confronta le etichette reali (y_test) con quelle previste (y_pred) e restituisce una matrice che mostra il numero di vere positivi, false positive, false negative e vere negative.
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good']) # 'sns.heatmap' crea una mappa di calore utilizzando i dati della matrice di confusione, 'annot=True' permette di annotare le celle con i valori numerici, 'fmt='d'' specifica il formato dei numeri come interi, 'cmap='Blues'' imposta la mappa dei colori su tonalità di blu, 'xticklabels' e 'yticklabels' impostano le etichette per gli assi x e y.
plt.ylabel('Valori Reali')                                                                                            # L'asse y rappresenta i valori reali delle classi (Bad o Good).
plt.xlabel('Valori Predetti')                                                                                         # L'asse y rappresenta i valori reali delle classi (Bad o Good).
plt.title('Matrice di Confusione')                                                                                    # Titolo che descrive il contenuto della mappa di calore.
plt.show()                                                                                                            # Visualizza la mappa di calore

# Storytelling sulle operazioni sulle mele (Matrice di Confusione)
print("""

MATRICE DI CONFUSIONE

La matrice di confusione è uno strumento fondamentale per valutare le prestazioni di un modello di classificazione. Essa è composta da quattro quadranti, ciascuno rappresentante un diverso risultato della 
classificazione:

- **True Negatives (TN)** - Alto a Sinistra: 528 (Bad-Bad)
- **False Positives (FP)** - Alto a Destra: 65 (Bad-Good)
- **False Negatives (FN)** - Basso a Sinistra: 67 (Good-Bad)
- **True Positives (TP)** - Basso a Destra: 540 (Good-Good)

### Interpretazione

1. **True Negatives (TN) = 528**
    - Il modello ha classificato correttamente 528 mele come cattive. Questo è un segnale positivo, indicando che il modello è in grado di riconoscere le mele di bassa qualità con un buon livello di affidabilità.

2. **False Positives (FP) = 65**
    - Qui, il modello ha erroneamente classificato 65 mele cattive come buone. Sebbene questo numero non sia eccessivamente alto, rappresenta un'area di miglioramento, poiché mele di bassa qualità potrebbero 
      essere vendute come buone, il che potrebbe portare a insoddisfazione dei clienti.

3. **False Negatives (FN) = 67**
    - Questo valore indica che il modello ha erroneamente classificato 67 mele buone come cattive. Ciò potrebbe avere un impatto negativo sulla produttività, poiché mele di buona qualità vengono scartate.

4. **True Positives (TP) = 540**
    - Il modello ha classificato correttamente 540 mele come buone. Questo risultato dimostra che il modello è capace di identificare con successo le mele di alta qualità.

### Report di Classificazione

Il report di classificazione presenta informazioni dettagliate sulle prestazioni del modello:

- **Precision**:
    - Bad: 0.89
    - Good: 0.89
    - La precisione indica la percentuale di classificazioni corrette tra le previste. Un valore di 0.89 suggerisce che il modello ha una buona precisione sia nella classificazione delle mele buone che cattive.

- **Recall**:
    - Bad: 0.89
    - Good: 0.89
    - Il richiamo misura la capacità del modello di identificare correttamente le classi positive. Anche in questo caso, un valore di 0.89 è rassicurante, poiché significa che il modello è in grado di 
      identificare la maggior parte delle mele buone e cattive.

- **F1-score**:
    - Bad: 0.89
    - Good: 0.89
    - L'F1-score è la media armonica tra precisione e richiamo, fornendo una misura complessiva dell'efficacia del modello. Valori intorno a 0.89 indicano un buon equilibrio tra precisione e recall.

- **Support**:
    - Bad: 593
    - Good: 607
    - Questo valore rappresenta il numero di istanze reali per ciascuna classe. È utile per comprendere il bilanciamento dei dati.

### Accuratezza

- **Accuracy: 0.89**
    - L'accuratezza del modello, pari a 0.89, indica che il 89% delle classificazioni effettuate dal modello sono corrette. Questo è un ottimo risultato e suggerisce che il modello è robusto.

### Media e Media Pesata

- **Macro Average: 0.89**
- **Weighted Average: 0.89**
    - Questi valori indicano che, anche se ci sono piccole differenze nel numero di campioni per ciascuna classe, il modello mantiene prestazioni coerenti attraverso entrambe le classi.

### Considerazioni Finali

In sintesi, il modello di classificazione ha dimostrato di avere prestazioni complessive buone e bilanciate. L'accuratezza del 89%, insieme ai valori di precisione, richiamo e F1-score, indica che il modello 
è efficace nel distinguere tra mele buone e cattive.
Tuttavia, è fondamentale monitorare e migliorare le aree evidenziate, come i falsi positivi e i falsi negativi, per ottimizzare ulteriormente le prestazioni del modello. L'implementazione di tecniche avanzate 
di apprendimento automatico potrebbe contribuire a perfezionare il modello, garantendo una classificazione più accurata e riducendo al minimo gli errori.

""")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convertiamo la colonna Date in formato datetime e creiamo una nuova colonna "Month" per aggregare per mese
sales_data['Date'] = pd.to_datetime(sales_data['Date'])                                                   # Converte la colonna 'Date' in formato datetime 
sales_data['Month'] = sales_data['Date'].dt.to_period('M')                                                # Estrae il mese e l'anno dalla data e lo memorizza in una nuova colonna 'Month'

# Raggruppiamo i dati per mese e sommiamo i profitti
monthly_sales = sales_data.groupby('Month')['Total'].sum().reset_index()                                  # Raggruppa i dati nel DataFrame 'sales_data' per la colonna 'Month', calcolando la somma delle vendite totali ('Total') per ciascun mese e il risultato viene poi reimpostato come un nuovo DataFrame con l'indice ripristinato.

# Trasformiamo la colonna Month in numerica per la regressione
monthly_sales['Month_numeric'] = monthly_sales['Month'].astype(str).str.replace('-', '').astype(int)      # Converte i periodi mensili dalla forma 'YYYY-MM' (stringa) a un formato numerico intero e sostituisce il trattino '-' con una stringa vuota e poi converte il risultato in un numero intero. Questo passaggio è utile per l'analisi di regressione, poiché la variabile indipendente deve essere numerica.

# Creiamo il modello di regressione
X = monthly_sales[['Month_numeric']]                                                                      # Seleziona la colonna 'Month_numeric' come feature (variabile indipendente) per il modello di regressione.
y = monthly_sales['Total']                                                                                # Seleziona la colonna 'Total' come target (variabile dipendente) che vogliamo prevedere.

# Dividiamo il dataset per il training e il test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)                 # Utilizza la funzione 'train_test_split' per dividere i dati in due set: uno per l'addestramento (80%) e uno per il test (20%).'random_state=42' assicura che la divisione sia riproducibile, fornendo sempre lo stesso set di dati per il training e il test.

# Inizializziamo il modello di regressione lineare
reg = LinearRegression()                                                                                  # Crea un'istanza del modello di regressione lineare.
reg.fit(X_train, y_train)                                                                                 # Addestra il modello sui dati di addestramento, apprendendo la relazione tra 'Month_numeric' e 'Total'.

# Facciamo previsioni sui dati di test
y_pred = reg.predict(X_test)                                                                              # Utilizza il modello di regressione addestrato (reg) per fare previsioni sui dati di test (X_test), producendo i valori previsti di vendita ('y_pred') per ciascun mese nel test set.

# Calcolo dell'errore quadratico medio per valutare la precisione del modello di regressione
mse = mean_squared_error(y_test, y_pred)                                                                  # Calcola l'errore quadratico medio (MSE), una metrica che misura quanto le previsioni differiscono dai valori reali. Confronta le previsioni 'y_pred' con i dati reali 'y_test'. Più basso è il valore, migliore è la precisione del modello.
print(f"\nMean Squared Error: {thousand_separator(mse)}")                                                 # Stampa il valore dell'errore quadratico medio formattato con il separatore delle migliaia, 'thousand_separator' applica il formato per la visualizzazione.

# Storytelling sull'errore quadratico medio
print("""
L'elevato valore di errore quadratico medio (MSE), può essere attribuito principalmente al fatto che i dati disponibili coprono solo un periodo di tre mesi. Questo limitato 
intervallo temporale non consente al modello di catturare in modo efficace eventuali trend stagionali, ciclici o variazioni nel comportamento delle vendite su periodi più lunghi.
""")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 
 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convertiamo 'Month' in stringa per la visualizzazione nel grafico
monthly_sales['Month'] = monthly_sales['Month'].dt.strftime('%Y-%m')                                                  # Converte la colonna 'Month' in formato stringa, nel formato 'YYYY-MM', per una migliore visualizzazione nel grafico.

# Visualizzazione del trend con un grafico a linea
plt.figure(figsize=(10, 6))                                                                                                 # Crea una nuova figura per il grafico con una dimensione specificata (larghezza, altezza).
plt.plot(monthly_sales['Month'], monthly_sales['Total'], marker='o', label='Vendite Totali', color='blue', linestyle='-')   # Crea un grafico a linee che rappresenta il trend delle vendite totali per mese, 'marker='o'' aggiunge dei marcatori per ciascun punto dati (vendite mensili), il colore è blu e le linee sono continue.
plt.xlabel('Mese')                                                                                                          # Imposta l'etichetta per l'asse X, rappresentando i mesi.
plt.ylabel('Vendite Totali')                                                                                                # Imposta l'etichetta per l'asse Y, rappresentando le vendite totali mensili.
plt.title('Trend delle Vendite Mensili nel Tempo')                                                                          # Imposta il titolo del grafico.
plt.xticks(rotation=45)                                                                                                     # Ruota le etichette dei mesi sull'asse X di 45 gradi per migliorarne la leggibilità.
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator_for_plot))                                             # Applica il separatore delle migliaia (formato europeo con punto) all'asse Y utilizzando la funzione 'FuncFormatter'.
plt.legend()                                                                                                                # Aggiunge una legenda per descrivere la linea nel grafico.
plt.grid()                                                                                                                  # Aggiunge una griglia al grafico per una migliore leggibilità visiva e per facilitare il confronto dei dati.

# Aggiungere etichette con i valori sopra i punti in grassetto
for i in range(len(monthly_sales)):                                             # Itera attraverso ogni riga del dataframe 'monthly_sales' utilizzando il suo indice, 'len(monthly_sales)' restituisce il numero di righe nel dataframe, in modo da poter accedere a ciascun punto dati.
    offset = monthly_sales['Total'][i] * - 0.01                                 # Definisce un piccolo offset per spostare leggermente l'etichetta sotto il punto, per evitare che si sovrapponga. L'offset è calcolato come l'1% del valore totale delle vendite, rendendolo dinamico rispetto all'altezza delle barre.
    plt.text(monthly_sales['Month'][i], monthly_sales['Total'][i] + offset,     # 'plt.text()' aggiunge del testo al grafico. Qui viene posizionato il valore delle vendite totali per ciascun mese. La posizione X corrisponde alla data (mese) e la posizione Y è il valore delle vendite totali con l'offset per spostare leggermente il testo.
             f"{thousand_separator(monthly_sales['Total'][i])}",                # formatta il valore delle vendite totali con un separatore delle migliaia ('.') per garantire che i numeri siano facilmente leggibili.
             ha='center', va='bottom', fontsize=9, fontweight='bold')           # 'ha' (horizontal alignment) allinea il testo orizzontalmente al centro del punto dati, 'va' (vertical alignment) allinea il testo verticalmente sul fondo del punto dati. Il 'fontsize=9' definisce una dimensione del testo appropriata per il grafico e 'fontweight='bold'' rende il testo in grassetto per migliorare la visibilità.

plt.tight_layout()                                                              # Assicura che gli elementi del grafico non si sovrappongano e che ci sia sufficiente spazio per tutte le etichette e assi.
plt.show()                                                                      # Mostra il grafico aggiornato con le etichette sopra i punti dati.

# Storytelling del trend delle Vendite Mensili nel Tempo
print("""
Il grafico mostra il trend delle vendite mensili per un periodo di tre mesi (gennaio, febbraio e marzo del 2019), con i seguenti valori totali di vendita:

Gennaio 2019: 338.942.184
Febbraio 2019: 269.491.740
Marzo 2019: 268.632.441

Osservazioni:
Calante dal primo al secondo mese:  
Dopo un forte inizio nel mese di gennaio, con vendite totali di circa 339 milioni, si nota un calo significativo a febbraio, con vendite scese a circa 269 milioni.

Stabilità tra febbraio e marzo:     
Dopo il calo tra gennaio e febbraio, le vendite si stabilizzano tra febbraio e marzo, con una leggera diminuzione di circa 860.000 unità, suggerendo che la flessione delle 
vendite potrebbe essersi arrestata.
    
Trend complessivo:                  
Complessivamente, il trend mostra un calo nelle vendite dal primo trimestre del 2019. Questo potrebbe essere dovuto a diversi fattori, come stagionalità, promozioni più forti 
all'inizio dell'anno o cicli economici del mercato.


Spiegazioni possibili:
Effetto stagionale:                             
È possibile che gennaio abbia beneficiato di eventi stagionali, come vendite post-festive, saldi o promozioni. I mesi successivi potrebbero non avere avuto 
lo stesso tipo di impulso.

Cambiamento nel comportamento dei consumatori:  
La flessione delle vendite potrebbe anche riflettere un cambiamento nei modelli di spesa dei clienti, che si stabilizzano dopo una spinta iniziale.
""")

print("-" * 40)                                         # Stampa una linea orizzontale di 40 caratteri 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
