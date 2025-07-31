#!/bin/bash

# Set the branch name (change this each time)
BRANCH_NAME="feature/feat_metadata"

# Switch to main and pull the latest changes
git checkout main && git pull origin main

# Create and switch to the new branch
git checkout -b $BRANCH_NAME

echo "✅ Branch $BRANCH_NAME created successfully"
echo "👉 Now, go to modify the code, and after completion, continue with the following commands:"

read -p "Press Enter to continue..."

# Add, commit, and push the changes
git add .
git commit -m "feat: add metadata module"
git push origin $BRANCH_NAME

echo "✅ Changes pushed to remote branch: $BRANCH_NAME"
echo "👉 Now, you can merge the branch into main"

read -p "Press Enter to continue merging to main..."

# Merge the branch into main
git checkout main
git merge $BRANCH_NAME
git push origin main

echo "✅ Branch merged and pushed to main successfully"
