## The aRxiv API package for R is way better than that for Python, hence I am using this one to scrape papers

library(aRxiv)
library(dplyr)

# categories is one of keys in categories.json
# both start_date and end_date are in format yyyymmdd

# which date can be "submitted", "updated", "both". 
# It sets which fields to look into for each paper to account for start_date and end_date. 
# "both" will check for the condition to be respected either in "submitted" OR in "updated"

retrieve_papers <- function(category, 
                            start_date, 
                            end_date, 
                            which_date="submitted", 
                            sort_by="submitted",
                            output_format="data.frame", 
                            batchsize=500, 
                            iter_sleep=10, 
                            csv_path=NULL) {
  
  if (which_date == "submitted") {
      date_range_query <- sprintf("submittedDate:[%s TO %s]", start_date, end_date)
    } else if (which_date == "updated") {
      date_range_query <- sprintf("lastUpdatedDate:[%s TO %s]", start_date, end_date)
    } else if (which_date == "both") {
      submitted <- sprintf("submittedDate:[%s TO %s]", start_date, end_date)
      updated <- sprintf("lastUpdatedDate:[%s TO %s]", start_date, end_date)
      date_range_query <- sprintf("(%s OR %s)", submitted, updated)
    } else {
      stop(sprintf("which_date only supports 'submitted', 'updated' or 'both', got %s", which_date))
  }
  
  category_query <- sprintf("cat:%s", category)
  final_query <- sprintf("%s AND %s", category_query, date_range_query)
  
  total_papers <- arxiv_count(query = final_query)
  print(sprintf("%s paper count for given date range: %s. Retrieving in portions of 1000 papers per iteration", category, total_papers))
  if (total_papers > 1000) {
    result <- arxiv_search(query = final_query, limit = 999, sort_by = sort_by,
                           batchsize = batchsize, start = 0, 
                           output_format = output_format, force = TRUE)
    iterations <- total_papers %/% 1000
    for (iter in 1:iterations) {
      print(sprintf("Sleeping for %s seconds before iteration nº %s", iter_sleep, iter))
      Sys.sleep(iter_sleep)
      partial_result <- arxiv_search(query = final_query, limit = 999, sort_by = sort_by, 
                                     batchsize = batchsize, start = 1000*iter, 
                                     output_format = output_format, force = TRUE)
      result <- bind_rows(result, partial_result)
    }
  } else {
    result <- arxiv_search(query = final_query, limit = NULL, sort_by = sort_by, 
                           batchsize = batchsize, output_format = output_format, force = TRUE)
  }
  
  final_df <- data.frame("arxiv_category" = rep(category, nrow(result)), "arxiv_abstract" = result$abstract)
  if (!is.null(csv_path)) {
    write.csv(final_df, csv_path, row.names = FALSE)
  } 
  return(final_df)
} 
