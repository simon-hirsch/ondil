# %%
library(dplyr)
library(tidyr)
if (!"RMySQL" %in% rownames(installed.packages())) {
  stop("Rymsql is needed.")
}
library(dbx) # Needs RMySQL as well...
library(RMySQL)
source("C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/playground/secrets.txt") # Load credentials (db_user, db_password)

# %% Establish a connection to the database
db <- dbxConnect(
  adapter = "mysql",
  host = "132.252.60.112",
  port = 3306,
  dbname = "DATASTREAM",
  user = db_user,
  password = db_password
)

# Check out the contents of the table
tbl(db, "datastream")

# Inspect the names of all products
unique(tbl(db, "datastream") |> pull(name))

# Get the data for a specific product in from a specific date
fuels <- tbl(db, "datastream") %>%
  filter(name %in% c(
    "coal_fM_01",
    "oil_fM_01", # USD
    #"gas_fD_01",
    "gas_fM_01",
    #"gas_fQ_01",
    #"gas_fY_01",
    #"EUA_spot",
    "EUA_fM_01",
    "USD_EUR"
  )) %>%
  filter(Date >= "2010-01-01") %>%
  select(-RIC) %>%
  collect() %>%
  pivot_wider(names_from = name, values_from = Value) %>%
  mutate(
    EUR_USD = 1 / USD_EUR,
    # Convert coal and oil prices to EUR
    across(starts_with("coal"), ~ round(.x / EUR_USD, 2)),
    across(starts_with("oil"), ~ round(.x / EUR_USD, 2)),
    Date = as.Date(Date) # ,
  ) %>%
  arrange(Date)


write.csv(fuels, "C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/fuels.csv", row.names = FALSE)
fuels
