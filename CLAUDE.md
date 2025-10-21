# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This project generates a coherent dataset about online gaming behavior for educational purposes, specifically for exploratory data analysis (EDA) teaching, and then applied Machine Learning (ML).

**Problem Statement**: The original Kaggle dataset ([Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset/data)) was poorly generated with no logical correlations or patterns between features, making it unsuitable for teaching EDA concepts although it was suitable for applied ML.

**Solution**: Create a synthetic dataset with similar features but with realistic, logical relationships and patterns that students can discover through analysis, which is also suitable for applied ML.

## Dataset Structure

### Original Dataset
The original dataset (`data/online_gaming_original.csv`) contains ~40,000 records with the following features:
- PlayerID, Age, Gender, Location, GameGenre, PlayTimeHours, InGamePurchases, GameDifficulty, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked, EngagementLevel

### Generated Dataset
The generated dataset (`generated_gaming_dataset.csv`) contains ~10,000 player-game combinations with 21 features organized as:

**Player-level attributes (fixed per PlayerID):**
- PlayerID, Age, Gender, Location

**Game-specific attributes:**
- GameID, GameName, GameGenre, GameDifficulty

**Behavioral & spending metrics:**
- PlayTimeHours, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked, EngagementLevel, DaysPlayed, PurchaseCount, TotalSpend, AvgPurchasesPerMonth, AvgPurchaseValue

**Target variables (for ML):**
- SpendingPropensity (NonSpender, Occasional, Whale)
- PlayerExpertise (Beginner, Intermediate, Expert)

## Project Goals

Generate a new dataset where:
1. Features have logical, natural correlations (e.g., PlayTimeHours correlates with PlayerLevel)
2. Patterns emerge that students can discover through EDA
3. Relationships reflect realistic gaming behavior
4. Data maintains statistical validity for educational exercises

## Architecture Notes

When implementing the data generation:
- The synthetic data should model realistic behavioral patterns (e.g., younger players might prefer Action games, higher engagement correlates with more achievements)
- Consider demographic influences on gaming preferences
- Build in meaningful correlations while avoiding overly deterministic relationships
- Ensure the data supports multiple types of analysis (correlation, segmentation, hypothesis testing)