#--------------------------------------------#
# INSTALLATION/MAJ DES LIRAIRIES NECESSAIRES #
#--------------------------------------------#

install.packages("nnet")
install.packages("ROCR")
install.packages("e1071")
install.packages("naivebayes")
install.packages("rpart")
install.packages("randomForest")
install.packages("kknn")
install.packages("C50")
install.packages("tree")
install.packages("dplyr")

#--------------------------------------#
# ACTIVATION DES LIRAIRIES NECESSAIRES #
#--------------------------------------#

library(nnet)
library(ROCR)
library(e1071)
library(naivebayes)
library(rpart)
library(randomForest)
library(kknn)
library(C50)
library(tree)
library(dplyr)


#-------------------------#
# PREPARATION DES DONNEES #
#-------------------------#

# Chargement des donnees
payment <- read.csv("Data Projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T)
payment_EA <- payment[1:800,]
payment_ET <- payment[801:1200,]
payment_to_predict <- read.csv("Data Projet New.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T)

#-----------------#
# NEURAL NETWORKS #
#-----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_nnet <- function(arg1, arg2, arg3, arg4, arg5){
  # Redirection de l'affichage des messages intermédiaires vers un fichier texte
  sink('output.txt', append=T)
  
  # Apprentissage du classifeur 
  nn <- nnet(default~., payment_EA, size = arg1, decay = arg2, maxit=arg3)
  
  # Réautoriser l'affichage des messages intermédiaires
  sink(file = NULL)
  
  # Test du classifeur : classe predite
  nn_class <- predict(nn, payment_ET, type="class")
  payment_ET$defaultPredicted <- nn_class
  
  # Matrice de confusion
   print(table(payment_ET$default, nn_class))
  
  # Test des classifeurs : probabilites pour chaque prediction
  nn_prob <- predict(nn, payment_ET, type="raw")
  print(nn_prob)
  
  # Courbe ROC 
  nn_pred <- prediction(nn_prob[,1], payment_ET$default)
  nn_perf <- performance(nn_pred,"tpr","fpr")
  plot(nn_perf, main = "Réseaux de neurones nnet()", add = arg4, col = arg5)
  
  # Calcul de l'AUC
  nn_auc <- performance(nn_pred, "auc")
  cat("AUC = ", as.character(attr(nn_auc, "y.values")))
  
  # Return ans affichage sur la console
  invisible()
}

#-------------------------#
# SUPPORT VECTOR MACHINES #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_svm <- function(arg1, arg2, arg3){
  # Apprentissage du classifeur
  svm <- svm(default~., payment_EA, probability=TRUE, kernel = arg1)
  
  # Test du classifeur : classe predite
  svm_class <- predict(svm, payment_ET, type="response")
  payment_ET$defaultPredicted <- svm_class
  
  # Matrice de confusion
  print(table(payment_ET$default, svm_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  svm_prob <- predict(svm, payment_ET, probability=TRUE)
  
  # Recuperation des probabilites associees aux predictions
  svm_prob <- attr(svm_prob, "probabilities")
  
  # Courbe ROC 
  svm_pred <- prediction(svm_prob[,1], payment_ET$default)
  svm_perf <- performance(svm_pred,"tpr","fpr")
  plot(svm_perf, main = "Support vector machines svm()", add = arg2, col = arg3)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  svm_auc <- performance(svm_pred, "auc")
  cat("AUC = ", as.character(attr(svm_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------#
# NAIVE BAYES #
#-------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_nb <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur 
  nb <- naive_bayes(default~., payment_EA, laplace = arg1, usekernel = arg2)
  
  # Test du classifeur : classe predite
  nb_class <- predict(nb, payment_ET, type="class")
  
  payment_ET$defaultPredicted <- nb_class
  
  # Matrice de confusion
  print(table(payment_ET$default, nb_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  nb_prob <- predict(nb, payment_ET, type="prob")
  
  # Courbe ROC
  nb_pred <- prediction(nb_prob[,2], payment_ET$default)
  nb_perf <- performance(nb_pred,"tpr","fpr")
  plot(nb_perf, main = "Classifieurs bayésiens naïfs naiveBayes()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  nb_auc <- performance(nb_pred, "auc")
  cat("AUC = ", as.character(attr(nb_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------------------#
# ARBRE DE DECISION RPART #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_rpart <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  dt <- rpart(default~., payment_EA, parms = list(split = arg1), control = rpart.control(minbucket = arg2))
  
  # Tests du classifieur : classe predite
  dt_class <- predict(dt, payment_ET, type="class")
  
  payment_ET$defaultPredicted <- dt_class
  
  # Matrice de confusion
  print(table(payment_ET$default, dt_class))
  
  # Tests du classifieur : probabilites pour chaque prediction
  dt_prob <- predict(dt, payment_ET, type="prob")
  
  # Courbes ROC
  dt_pred <- prediction(dt_prob[,2], payment_ET$default)
  dt_perf <- performance(dt_pred,"tpr","fpr")
  plot(dt_perf, main = "Arbres de décision rpart()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  dt_auc <- performance(dt_pred, "auc")
  cat("AUC = ", as.character(attr(dt_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#----------------#
# RANDOM FORESTS #
#----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_rf <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  rf <- randomForest(default~., payment_EA, ntree = arg1, mtry = arg2)
  
  # Test du classifeur : classe predite
  rf_class <- predict(rf,payment_ET, type="response")
  payment_ET$defaultPredicted <- rf_class
  
  # Matrice de confusion
  print(table(payment_ET$default, rf_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  rf_prob <- predict(rf, payment_ET, type="prob")
  
  # Courbe ROC
  rf_pred <- prediction(rf_prob[,2], payment_ET$default)
  rf_perf <- performance(rf_pred,"tpr","fpr")
  plot(rf_perf, main = "Random Forests randomForest()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  rf_auc <- performance(rf_pred, "auc")
  cat("AUC = ", as.character(attr(rf_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#---------------------#
# K-NEAREST NEIGHBORS #
#---------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_knn <- function(arg1, arg2, arg3, arg4){
  # Apprentissage et test simultanes du classifeur de type k-nearest neighbors
  knn <- kknn(default~., payment_EA, payment_ET, k = arg1, distance = arg2)
  
  # Matrice de confusion
  print(table(payment_ET$default, knn$fitted.values))
  
  # Courbe ROC
  knn_pred <- prediction(knn$prob[,2], payment_ET$default)
  knn_perf <- performance(knn_pred,"tpr","fpr")
  plot(knn_perf, main = "Classifeurs K-plus-proches-voisins kknn()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  knn_auc <- performance(knn_pred, "auc")
  cat("AUC = ", as.character(attr(knn_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

test_C50 <- function()
{
  # Apprentissage arbre sur 'produit_EA'
  tree1 <- C5.0(default~., payment_EA)
  
  # Affichages graphiques 
  plot(tree1, type="simple")

  # Application de 'tree1' sur l'ensemble de test produit_ET
  test_tree1 <- predict(tree1, payment_ET, type="class")
  
  # Matrice de confusion des tests de 'tree1' sur l'ensemble de test produit_ET
  mc_tree1 <- table(payment_ET$default, test_tree1)
  print(mc_tree1)
}

#----------------------------------------#
# APPRENTISSAGE ARBRE DE DECISION 'tree' #
#----------------------------------------#

test_tree <- function()
{
  # Apprentissage arbre
tree3 <- tree(default~., data=payment_EA)

# Affichage graphique : tracage des arcs par la fonction plot.tree() 
plot(tree3)

# Application de 'tree1' sur l'ensemble de test produit_ET
test_tree1 <- predict(tree3, payment_ET, type="class")
mc_tree1 <- table(payment_ET$default, test_tree1)
print(mc_tree1)

# Ajout du texte au graphique par la fonction text.tree()
text(tree3, pretty=0)
}



#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

test_C50()
test_tree()

# Réseaux de neurones nnet()
test_nnet(50, 0.01, 100, FALSE, "red")
test_nnet(50, 0.01, 300, TRUE, "tomato")
test_nnet(25, 0.01, 100, TRUE, "blue")
test_nnet(25, 0.01, 300, TRUE, "purple")
test_nnet(50, 0.001, 100, TRUE, "green")
test_nnet(50, 0.001, 300, TRUE, "turquoise")
test_nnet(25, 0.001, 100, TRUE, "grey")
test_nnet(25, 0.001, 300, TRUE, "black")

# Support vector machines
test_svm("linear", FALSE, "red")
test_svm("polynomial", TRUE, "blue")
test_svm("radial", TRUE, "green")
test_svm("sigmoid", TRUE, "orange")

# Naive Bayes
test_nb(0, FALSE, FALSE, "red")
test_nb(20, FALSE, TRUE, "blue")
test_nb(0, TRUE, TRUE, "green")
test_nb(20, TRUE, TRUE, "orange")

# Arbres de decision
test_rpart("gini", 10, FALSE, "red")
test_rpart("gini", 5, TRUE, "blue")
test_rpart("information", 10, TRUE, "green")
test_rpart("information", 5, TRUE, "orange")

# Forets d'arbres decisionnels aleatoires
test_rf(300, 3, FALSE, "red")
test_rf(300, 5, TRUE, "blue")
test_rf(500, 3, TRUE, "green")
test_rf(500, 5, TRUE, "orange")

# K plus proches voisins
test_knn(10, 1, FALSE, "red")
test_knn(10, 2, TRUE, "blue")
test_knn(20, 1, TRUE, "green")
test_knn(20, 2, TRUE, "orange")


#-------------------------------------------------#
#  APPLICATION DU CLASSIFIEUR A DATA PROJET NEW   #
#-------------------------------------------------#

formatProba <- function(x){
  if(x<=0.5){
    return(1-x)
  } else {
    return(x)
  }
}

predictPayment <- function(arg1, arg2, arg3){
  
  # Création du modèle
  nn <- nnet(default~., payment_EA, size = arg1, decay = arg2, maxit=arg3)
  
  # Application du modèle à data projet new
  nn_class <- predict(nn, payment_to_predict, type="class")
  
  # Calcul des probabilités de chaque prédiction dans data projet new
  nn_prob <- predict(nn, payment_to_predict, type="raw")
  
  print(nn_class)
  print(nn_prob)
  
  payment_predicted <- select(payment_to_predict, customer)
  payment_predicted$default <- nn_class
  payment_predicted$prob <- sapply(nn_prob[,1], formatProba)
  
  print(payment_predicted)
  
  write.csv(payment_predicted, './Data projet new predicted.csv', row.names = FALSE)
  
  # Return ans affichage sur la console
  invisible()
}

predictPayment(25, 0.01, 300)

