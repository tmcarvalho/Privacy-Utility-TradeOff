library(ggplot2)

reID <- read.csv("tilemap.csv", header=TRUE, sep='\t')


reID$ds <- factor(reID$ds, levels = rev(unique(reID$ds)), ordered = TRUE)
reID$char <- as.character(reID$comb)
reID$char <- nchar(reID$char)
reID <- reID[order(reID$char), ]
reID$comb <- factor(reID$comb, levels = unique(reID$comb), ordered = TRUE)
reID$text <- format(round(reID$rl, 0))
#colnames(reID)[4] <- "Record Linkage"
#colnames(reID)[5] <- "Rank"

p <- ggplot(reID,aes(x=comb, y=ds, fill=rl))+
  geom_tile(colour="white",size=0.25)+
  geom_text(aes(label = round(rl)),color='white', size=2) + 
  #add border white colour of line thickness 0.25
  # geom_text(aes(label=as.factor(text)),size=2)+
  #remove x and y axis labels
  labs(x="",y="", fill="Percentage of\nre-identification")+
  #remove extra space
  # scale_y_discrete(expand=c(0,0))+
  #reverse scale
  scale_fill_continuous(trans = 'reverse')+
  # scale_fill_discrete(name="Record Linkage")+
  #theme options
  theme(
    #bold font for legend text
    legend.text=element_text(face="bold"),
    #set thickness of axis ticks
    axis.ticks=element_line(size=0.4),
    #remove plot background
    plot.background=element_blank(),
    #remove plot border
    panel.border=element_blank(),
    #rotate x labels
    axis.text.x=element_text(angle=45,hjust=1))

print(p)


r <- ggplot(reID,aes(x=comb,y=ds,fill=rank))+
  #geom_tile() +
  #add border white colour of line thickness 0.25
  geom_tile(colour="white",size=0.25)+
  #remove x and y axis labels
  labs(x="",y="", fill="Rank of\nre-identification")+
  #remove extra space
  scale_y_discrete(expand=c(0,0))+
  geom_text(aes(label = round(rank)),color='white', size=2) + 
  #reverse scale
  scale_fill_continuous(trans = 'reverse')+
  #theme options
  theme(
    #bold font for legend text
    legend.text=element_text(face="bold"),
    #set thickness of axis ticks
    axis.ticks=element_line(size=0.4),
    #remove plot background
    plot.background=element_blank(),
    #remove plot border
    panel.border=element_blank(),
    #rotate x labels
    axis.text.x=element_text(angle=45,hjust=1))

print(r)
