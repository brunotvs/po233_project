library(dplyr)
library(ggplot2)
library(reshape2)
library(PerformanceAnalytics)

rm(list = ls())

setwd("C:/Users/walla/OneDrive/Área de Trabalho/Machine")
df <- read.delim("dataset_merge.txt", sep=",")

df[,c(2:3,8:11)] <- NULL

df$Data <- as.Date(df$Data)
df$river <- as.character(df$river)

#Quantidade de linhas por rio
values <- df %>%
  group_by(river) %>%
  summarise(count=n())
values$Perc <- values$count/sum(values$count)*100

#Dispersão da média no Rio 1
df2 <- subset(df, Data >= as.Date("1999-01-01") & Data <= as.Date("1999-01-10") & river==c("1"))

ggplot(df2, aes(x = Data, y = PREC, color = Data, group = Data)) +
  geom_boxplot(position = position_dodge(width = 0.2)) +
  geom_point(position = position_jitterdodge(seed = 123))

#Dispersão da média entre Rios
df2 <- subset(df, Data >= as.Date("1999-01-01") & Data <= as.Date("1999-01-01"))

ggplot(df2, aes(x = river, y = PREC, color = river, group = river)) +
  geom_boxplot(position = position_dodge(width = 0.2)) +
  geom_point(position = position_jitterdodge(seed = 123))

#Caso usássemos a soma das precipitações
teste <- df2  %>%
  group_by(Data,river) %>%
  summarize_all((sum))

ggplot(teste, aes(x=river, y=PREC, fill=PREC)) +
  geom_bar(stat="identity")+theme_minimal()

#Caso usássemos a média das precipitações
teste <- df2  %>%
  group_by(Data,river) %>%
  summarize_all((mean))

ggplot(teste, aes(x=river, y=PREC, fill=PREC)) +
  geom_bar(stat="identity")+theme_minimal()

#Usando média para tudo
# teste2 <- subset(df, Data >= as.Date("1999-01-01") & Data <= as.Date("1999-01-02"))
# teste2 <- subset(df, Data >= as.Date("1999-01-01") & Data <= as.Date("1999-01-01") & river==c("1"))

teste <- df %>%
  group_by(Data,river)%>%
  mutate(PREC=sum(PREC)) %>%
  mutate(evap=sum(evap)) %>%
  mutate(temp=mean(temp)) %>%
  mutate(rnof=mean(rnof))
teste <- teste %>%
  slice(1)

#Distribuição das precipitações em 30 dias
df2 <- subset(teste, Data >= as.Date("1999-03-01") & Data <= as.Date("1999-03-30"))

ggplot(df2, aes(x = river, y = PREC-evap, color = river, group = river)) +
  geom_boxplot(position = position_dodge(width = 0.2)) +
  geom_point(position = position_jitterdodge(seed = 123))

teste$contribution <- NULL
teste$river <- gsub("A",9,teste$river)
teste$river <- gsub("B",10,teste$river)
teste$river <- gsub("C",11,teste$river)
teste$river <- as.numeric(teste$river)

nivel <- read.delim("Nivel.txt", sep=",")
nivel$Data <- as.Date(nivel$Data, format="%m/%d/%Y")
vazao <- read.delim("Vazao.txt", sep=",")
vazao$Data <- as.Date(vazao$Data, format="%Y-%m-%d")

nivel$ReservatÃ³rio <- NULL
vazao$ReservatÃ³rio <- NULL

teste2 <- merge(teste,nivel, by=c("Data"))
teste2 <- merge(teste2,vazao, by=c("Data"))

teste2 <- teste2[complete.cases(teste2), ]

teste3 <- subset(teste2, river==c("10"))
chart.Correlation(teste3[,-c(1,6)], histogram=TRUE, pch=19)



