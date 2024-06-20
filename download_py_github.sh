#!/usr/bin/env bash

# Read the GitHub token from the configuration file
GH_TOKEN=$(<~/.gh_token)

# Ensure GH_TOKEN is set
if [ -z "$GH_TOKEN" ]; then
    echo "Error: GitHub token is not set in ~/.gh_token."
    exit 1
fi

# Set GitHub API base URL for searching Python files
base_url="search/code?q=extension:py+sklearn+in:file&per_page=100"

# Initialize variables
page=1
total_results=()
total_count=0

# Fetch search results from GitHub using the PAT for authentication and paginate
while [ $total_count -lt 400 ]; do
    api_url="$base_url&page=$page"
    response=$(GH_TOKEN=$GH_TOKEN gh api --method=GET "$api_url" -H "Authorization: token $GH_TOKEN")

    # Check if the response contains items
    if echo "$response" | jq -e '.items' > /dev/null; then
        items=$(echo "$response" | jq -r '.items[].html_url')
        count=$(echo "$items" | wc -l)
        
        # Append items to total_results
        total_results+=($items)
        total_count=$((total_count + count))
        
        # Break if fewer than 100 items were fetched (no more pages)
        if [ $count -lt 100 ]; then
            break
        fi
    else
        echo "No items found or API rate limit exceeded."
        exit 1
    fi

    # Increment the page number
    page=$((page + 1))
done

# Create a directory to store the downloaded files
mkdir -p downloaded_files
cd downloaded_files

# Download all files
for url in "${total_results[@]}"; do
    raw_url=$(echo $url | sed 's/github.com/raw.githubusercontent.com/' | sed 's/blob\///')
    curl -O $raw_url
done

echo "Download completed."


