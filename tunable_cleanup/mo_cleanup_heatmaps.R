#---
#title: "mo_heatmaps"
#author: "David O'Callaghan"
#date: "7/31/2020"
#output: html_document
#---

#```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
#```


#```{r message=FALSE, warning=FALSE}
data <- read_csv('./tunable-agents/tunable_cleanup/results/cleanup_tuning_varied_prefs_210421.csv')

ggplot(data) +
  geom_tile(aes(x=1 - `Competitive 1`/0.4, y=1 - `Competitive 2`/0.4, fill=`Agent 2 Clean`/250)) +
  scale_x_continuous(breaks=seq(0,1,1/5), name="Agent 1 Cooperativeness", expand = c(0,0)) +
  scale_y_continuous(breaks=seq(0,1,1/5), name="Agent 2 Cooperativeness", expand = c(0,0)) +

  scale_fill_distiller(type='seq', palette=1, direction = 1, name="Agent 1 Cleaning Rate",
                       breaks=seq(0.3,0.8,0.1),
                       limits=c(0.248,0.84)) +

  coord_fixed() +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.key.size = unit(10, "mm"),
    legend.position = "right",

    axis.title = element_text(size=12),
  )

ggsave('./tunable-agents/tunable_cleanup/results/cleanup_heatmap210421_agent2.png')
#```
