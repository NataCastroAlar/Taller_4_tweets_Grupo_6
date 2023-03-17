
###GRUPO 6###
###VICTOR CHIQUE, VICTOR SANHEZ, NATALIA CASTRO

rm(list = ls())

library(pacman) 

# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       tm,   # para Text Mining
       tidytext, #Para tokenización
       wordcloud, # Nube de palabras 
       SentimentAnalysis, #Análisis de sentimientos,
       topicmodels
) 


train <- read_csv("Documents/UNIANDES BIG DATA MACHINE LEARNING/Archivos R/DATOS TALLERES/DATOS POLITICOS/train.csv")
test <- read_csv("Documents/UNIANDES BIG DATA MACHINE LEARNING/Archivos R/DATOS TALLERES/DATOS POLITICOS/test.csv")

comentarios<-train$text

p_load("stringi")
comentarios <- stri_trans_general(str = train$text, id = "Latin-ASCII") #elimnar tíldes
comentarios[1]
substr(comentarios, 1, 100)

comentarios <- gsub('[^A-Za-z0-9 ]+', ' ', comentarios) #eliminamos todo lo que no sea alfanumérico
substr(comentarios, 1, 100)


comentarios <- tolower(comentarios)# Minúsculas
substr(comentarios, 1, 100)


comentarios <- gsub('\\s+', ' ', comentarios) # Espacios
substr(comentarios, 1, 100)

##Como nos interesan sólo las palabras elimnamos los números y espacios no simples
comentarios <- gsub("\\d+", "", comentarios)
comentarios <- gsub('\\s+', ' ', comentarios)
comentarios <- trimws(comentarios)
substr(comentarios, 1, 100)


comentarios <- iconv(comentarios, from = "UTF-8", to="ASCII//TRANSLIT")
comentarios[1]

wordcloud(comentarios, max.words = 300, colors = brewer.pal(8, "Paired"))

##-------------------------------------------------------------------------------
#Convertimos comentarios en corpus

texts<-VCorpus(VectorSource(comentarios))
texts[1]

#Eliminamos stopwords en español y elimanos las tildes de ese menú poque ya se las quité al mio
stopwords_2 <- iconv(stopwords("spanish"), from = "UTF-8", to="ASCII//TRANSLIT")
texts = tm_map (texts, removeWords, stopwords_2)

## stemming

texts = tm_map(texts, stemDocument, language = "spanish")

wordcloud(texts, max.words = 300, colors = brewer.pal(8, "Paired"))


###CREAMOS MATRIZ---------------------------------------------------------------
matriz_1 = DocumentTermMatrix(texts)
inspect(matriz_1[1:100,])
matriz_1 <- removeSparseTerms(matriz_1, sparse = 0.95)
inspect(matriz_1[1:100,])

##CONVERTIMOS LA MATRIZ EN DATA FRAME-------------------------------------------
matriz_df = as.data.frame(as.matrix(matriz_1)) #convierte en data frame
colnames(matriz_df) = make.names(colnames(matriz_df)) #variables R friendly
matriz_df$name = train$name #añade la variable dependiente al set de datos

## Porentaje Uribe - Lopez - Petro
prop.table(table(matriz_df$name))


### CREAMOS TRAIN Y TEST--------------------------------------------------------

set.seed(123)
train_index <- createDataPartition(matriz_df$name, p = .8)$Resample1
train_df <- matriz_df[train_index,]
test_df <- matriz_df[-train_index,]

###MODELO RANDOM FOREST---------------------------------------------------------

mtry_grid<-expand.grid(mtry =c(8,10,12))
ctrl<- trainControl(method = "cv",
                    number = 5)


p_load(randomForest)
set.seed(123)
forest <- train(name~., 
                data = train_df, 
                method = "rf",
                trControl = ctrl,
                tuneGrid=mtry_grid,
                metric="Accuracy",
                ntree=200
)

forest

plot(forest)
