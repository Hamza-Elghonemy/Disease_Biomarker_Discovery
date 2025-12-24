library(readxl)  # for reading Excel files
library(ggplot2)
library(dplyr)
library(tidyr)
library(pheatmap)
library(RColorBrewer)
library(FactoMineR)
library(factoextra)

# Load data from Excel
file_path <- file.choose()  # Choose the Excel file interactively
data <- read_excel(file_path, sheet = 1)  # Reads the first sheet
data <- as.data.frame(data)  # Convert to data.frame if needed
rownames(data) <- data[[1]]  # Assuming first column is row names
data <- data[,-1]  # Remove the first column after setting rownames

# Detect label column if it exists
label_col <- intersect(c("label", "group", "Group", "class", "Class"), colnames(data))

if (length(label_col) == 1) {
  labels <- as.factor(data[[label_col]])
  features <- data %>% select(-all_of(label_col))
} else {
  labels <- NULL
  features <- data
}

# Top 20 by variance
var_features <- apply(features, 2, var, na.rm = TRUE)
top20_features <- names(sort(var_features, decreasing = TRUE))[1:min(20, ncol(features))]
features_top20 <- features[, top20_features]

# Global distribution
ggplot(data.frame(value = as.vector(as.matrix(features_top20))),
       aes(x = value)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
  labs(title = "Raw Value Distribution (Top 20 Variable Features)",
       x = "Raw value", y = "Frequency") +
  theme_minimal()

# Boxplot
boxplot(features_top20, outline = FALSE, col = "lightgray",
        main = "Sample-wise Distribution (Top 20, Raw Data)",
        ylab = "Raw values", las = 2)

# Missing values
missing_pct <- colMeans(is.na(features_top20)) * 100
ggplot(data.frame(Feature = names(missing_pct), Missing = missing_pct),
       aes(x = reorder(Feature, Missing), y = Missing)) +
  geom_bar(stat = "identity", fill = "darkred") +
  coord_flip() +
  labs(title = "Missing Values (Top 20 Features)",
       x = "Feature", y = "% Missing") +
  theme_minimal()

# PCA
pca <- PCA(features_top20, scale.unit = FALSE, graph = FALSE)

if (!is.null(labels)) {
  fviz_pca_ind(pca, col.ind = labels, addEllipses = TRUE,
               legend.title = "Group",
               title = "PCA – Raw Data (Top 20)")
} else {
  fviz_pca_ind(pca, title = "PCA – Raw Data (Top 20)")
}

# Heatmap
annotation_row <- NULL
if (!is.null(labels)) {
  annotation_row <- data.frame(Group = labels)
  rownames(annotation_row) <- rownames(features_top20)
}

pheatmap(features_top20,
         scale = "none",
         show_rownames = FALSE,
         annotation_row = annotation_row,
         color = colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
         main = "Heatmap of Top 20 Variable Features (Raw Data)")

# Density plots (Top 5)
features_top20 %>%
  select(1:min(5, ncol(features_top20))) %>%
  pivot_longer(cols = everything()) %>%
  ggplot(aes(x = value, color = name)) +
  geom_density() +
  labs(title = "Density Plot of Top Variable Features",
       x = "Raw value", y = "Density") +
  theme_minimal()

# Histograms for Top 5 Features
features_top20 %>%
  select(1:min(5, ncol(features_top20))) %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Feature)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_wrap(~Feature, scales = "free") +
  labs(title = "Histograms of Top Variable Features (Raw Data)",
       x = "Raw value", y = "Count") +
  theme_minimal()
