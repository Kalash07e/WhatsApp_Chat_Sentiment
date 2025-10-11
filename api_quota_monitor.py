"""
API Usage Tracker Module
========================

This module provides comprehensive tracking and monitoring for Google Gemini API usage.
It helps users monitor their quota limits, usage patterns, and get warnings when approaching limits.

Features:
- Real-time quota tracking
- Daily/monthly usage statistics
- Usage history and patterns
- Automatic warnings and alerts
- Admin dashboard integration
- Quota reset functionality

Author: Kalash Bhargava
Date: October 4, 2025
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import streamlit as st


class APIUsageTracker:
    """
    Comprehensive API usage tracking and monitoring system
    """
    
    def __init__(self, tracker_file: str = "api_usage_tracker.json"):
        """
        Initialize the API usage tracker
        
        Args:
            tracker_file (str): Path to the JSON file storing usage data
        """
        self.tracker_file = tracker_file
        self.data = self._load_usage_data()
        
    def _load_usage_data(self) -> Dict[str, Any]:
        """
        Load usage data from JSON file or create default structure
        
        Returns:
            Dict containing usage tracking data
        """
        try:
            if os.path.exists(self.tracker_file):
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            else:
                return self._create_default_structure()
        except Exception as e:
            st.error(f"Error loading usage data: {str(e)}")
            return self._create_default_structure()
    
    def _create_default_structure(self) -> Dict[str, Any]:
        """
        Create default data structure for API usage tracking
        
        Returns:
            Dict with default tracking structure
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "google_gemini_api": {
                "total_quota_limit": 250,
                "current_usage": {
                    "daily": 0,
                    "monthly": 0,
                    "total": 0
                },
                "remaining_quota": 250,
                "usage_history": {
                    "daily_usage": {},
                    "monthly_usage": {}
                },
                "last_reset_date": today,
                "last_usage_date": None,
                "quota_warnings": {
                    "75_percent_warning": False,
                    "90_percent_warning": False,
                    "95_percent_warning": False
                },
                "usage_stats": {
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "average_daily_usage": 0,
                    "peak_usage_day": None,
                    "peak_usage_count": 0
                },
                "api_key_info": {
                    "created_date": today,
                    "expires_date": None,
                    "status": "active"
                }
            },
            "tracking_metadata": {
                "file_version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "tracking_enabled": True,
                "auto_reset_monthly": True
            }
        }
    
    def _save_usage_data(self) -> None:
        """
        Save usage data to JSON file
        """
        try:
            self.data["tracking_metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.tracker_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving usage data: {str(e)}")
    
    def record_api_usage(self, success: bool = True) -> None:
        """
        Record an API usage event
        
        Args:
            success (bool): Whether the API call was successful
        """
        today = datetime.now().strftime("%Y-%m-%d")
        current_month = datetime.now().strftime("%Y-%m")
        
        # Update current usage
        self.data["google_gemini_api"]["current_usage"]["total"] += 1
        self.data["google_gemini_api"]["current_usage"]["daily"] += 1
        self.data["google_gemini_api"]["current_usage"]["monthly"] += 1
        
        # Update remaining quota
        self.data["google_gemini_api"]["remaining_quota"] = max(
            0, 
            self.data["google_gemini_api"]["total_quota_limit"] - 
            self.data["google_gemini_api"]["current_usage"]["total"]
        )
        
        # Update usage history
        if today not in self.data["google_gemini_api"]["usage_history"]["daily_usage"]:
            self.data["google_gemini_api"]["usage_history"]["daily_usage"][today] = 0
        self.data["google_gemini_api"]["usage_history"]["daily_usage"][today] += 1
        
        if current_month not in self.data["google_gemini_api"]["usage_history"]["monthly_usage"]:
            self.data["google_gemini_api"]["usage_history"]["monthly_usage"][current_month] = 0
        self.data["google_gemini_api"]["usage_history"]["monthly_usage"][current_month] += 1
        
        # Update success/failure stats
        if success:
            self.data["google_gemini_api"]["usage_stats"]["successful_requests"] += 1
        else:
            self.data["google_gemini_api"]["usage_stats"]["failed_requests"] += 1
        
        # Update peak usage tracking
        daily_usage = self.data["google_gemini_api"]["usage_history"]["daily_usage"][today]
        if daily_usage > self.data["google_gemini_api"]["usage_stats"]["peak_usage_count"]:
            self.data["google_gemini_api"]["usage_stats"]["peak_usage_count"] = daily_usage
            self.data["google_gemini_api"]["usage_stats"]["peak_usage_day"] = today
        
        # Update last usage date
        self.data["google_gemini_api"]["last_usage_date"] = today
        
        # Check and update quota warnings
        self._check_quota_warnings()
        
        # Calculate average daily usage
        self._calculate_average_usage()
        
        # Save updated data
        self._save_usage_data()
    
    def _check_quota_warnings(self) -> None:
        """
        Check and update quota warning flags
        """
        usage_percentage = (
            self.data["google_gemini_api"]["current_usage"]["total"] / 
            self.data["google_gemini_api"]["total_quota_limit"]
        ) * 100
        
        if usage_percentage >= 75:
            self.data["google_gemini_api"]["quota_warnings"]["75_percent_warning"] = True
        if usage_percentage >= 90:
            self.data["google_gemini_api"]["quota_warnings"]["90_percent_warning"] = True
        if usage_percentage >= 95:
            self.data["google_gemini_api"]["quota_warnings"]["95_percent_warning"] = True
    
    def _calculate_average_usage(self) -> None:
        """
        Calculate average daily usage
        """
        daily_usage = self.data["google_gemini_api"]["usage_history"]["daily_usage"]
        if daily_usage:
            total_days = len(daily_usage)
            total_usage = sum(daily_usage.values())
            self.data["google_gemini_api"]["usage_stats"]["average_daily_usage"] = round(
                total_usage / total_days, 2
            )
    
    def get_quota_status(self) -> Dict[str, Any]:
        """
        Get current quota status and statistics
        
        Returns:
            Dict containing quota status information
        """
        total_used = self.data["google_gemini_api"]["current_usage"]["total"]
        total_limit = self.data["google_gemini_api"]["total_quota_limit"]
        usage_percentage = (total_used / total_limit) * 100
        
        return {
            "total_quota": total_limit,
            "used_quota": total_used,
            "remaining_quota": self.data["google_gemini_api"]["remaining_quota"],
            "usage_percentage": round(usage_percentage, 2),
            "daily_usage": self.data["google_gemini_api"]["current_usage"]["daily"],
            "monthly_usage": self.data["google_gemini_api"]["current_usage"]["monthly"],
            "quota_status": self._get_quota_status_text(usage_percentage),
            "warnings_active": any(self.data["google_gemini_api"]["quota_warnings"].values())
        }
    
    def _get_quota_status_text(self, usage_percentage: float) -> str:
        """
        Get human-readable quota status
        
        Args:
            usage_percentage (float): Current usage percentage
            
        Returns:
            String describing quota status
        """
        if usage_percentage >= 95:
            return "🔴 Critical - Almost Exhausted"
        elif usage_percentage >= 90:
            return "🟠 Warning - High Usage"
        elif usage_percentage >= 75:
            return "🟡 Caution - Moderate Usage"
        elif usage_percentage >= 50:
            return "🔵 Normal - Moderate Usage"
        else:
            return "🟢 Excellent - Low Usage"
    
    def get_usage_warnings(self) -> List[str]:
        """
        Get list of active usage warnings
        
        Returns:
            List of warning messages
        """
        warnings = []
        quota_status = self.get_quota_status()
        
        if self.data["google_gemini_api"]["quota_warnings"]["95_percent_warning"]:
            warnings.append("🚨 CRITICAL: 95% of API quota used! Consider upgrading or reducing usage.")
        elif self.data["google_gemini_api"]["quota_warnings"]["90_percent_warning"]:
            warnings.append("⚠️ WARNING: 90% of API quota used! Monitor usage carefully.")
        elif self.data["google_gemini_api"]["quota_warnings"]["75_percent_warning"]:
            warnings.append("⚡ NOTICE: 75% of API quota used. Plan usage for rest of the period.")
        
        if quota_status["remaining_quota"] == 0:
            warnings.append("🛑 API quota exhausted! Threat analysis is temporarily disabled.")
        
        return warnings
    
    def can_make_api_call(self) -> Tuple[bool, str]:
        """
        Check if API call can be made based on remaining quota
        
        Returns:
            Tuple of (can_make_call, reason)
        """
        if self.data["google_gemini_api"]["remaining_quota"] <= 0:
            return False, "API quota exhausted"
        
        if not self.data["tracking_metadata"]["tracking_enabled"]:
            return False, "API tracking disabled"
        
        return True, "API call allowed"
    
    def get_usage_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Get usage history for specified number of days
        
        Args:
            days (int): Number of days to include in history
            
        Returns:
            Dict containing usage history data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        daily_history = {}
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            daily_history[date_str] = self.data["google_gemini_api"]["usage_history"]["daily_usage"].get(date_str, 0)
            current_date += timedelta(days=1)
        
        return {
            "daily_history": daily_history,
            "total_period_usage": sum(daily_history.values()),
            "average_daily": round(sum(daily_history.values()) / len(daily_history), 2),
            "peak_day": max(daily_history.items(), key=lambda x: x[1]) if daily_history else (None, 0)
        }
    
    def reset_quota(self, reset_type: str = "monthly") -> bool:
        """
        Reset quota counters
        
        Args:
            reset_type (str): Type of reset - 'daily', 'monthly', or 'total'
            
        Returns:
            bool: Success status
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            if reset_type == "daily":
                self.data["google_gemini_api"]["current_usage"]["daily"] = 0
            elif reset_type == "monthly":
                self.data["google_gemini_api"]["current_usage"]["monthly"] = 0
                self.data["google_gemini_api"]["current_usage"]["total"] = 0
                self.data["google_gemini_api"]["remaining_quota"] = self.data["google_gemini_api"]["total_quota_limit"]
                self.data["google_gemini_api"]["last_reset_date"] = today
                # Reset warnings
                for warning in self.data["google_gemini_api"]["quota_warnings"]:
                    self.data["google_gemini_api"]["quota_warnings"][warning] = False
            elif reset_type == "total":
                self.data["google_gemini_api"]["current_usage"] = {"daily": 0, "monthly": 0, "total": 0}
                self.data["google_gemini_api"]["remaining_quota"] = self.data["google_gemini_api"]["total_quota_limit"]
                self.data["google_gemini_api"]["usage_history"] = {"daily_usage": {}, "monthly_usage": {}}
                self.data["google_gemini_api"]["usage_stats"]["successful_requests"] = 0
                self.data["google_gemini_api"]["usage_stats"]["failed_requests"] = 0
                self.data["google_gemini_api"]["last_reset_date"] = today
                # Reset warnings
                for warning in self.data["google_gemini_api"]["quota_warnings"]:
                    self.data["google_gemini_api"]["quota_warnings"][warning] = False
            
            self._save_usage_data()
            return True
        except Exception as e:
            st.error(f"Error resetting quota: {str(e)}")
            return False
    
    def update_quota_limit(self, new_limit: int) -> bool:
        """
        Update the total quota limit
        
        Args:
            new_limit (int): New quota limit
            
        Returns:
            bool: Success status
        """
        try:
            self.data["google_gemini_api"]["total_quota_limit"] = new_limit
            self.data["google_gemini_api"]["remaining_quota"] = max(
                0, 
                new_limit - self.data["google_gemini_api"]["current_usage"]["total"]
            )
            self._save_usage_data()
            return True
        except Exception as e:
            st.error(f"Error updating quota limit: {str(e)}")
            return False
    
    def export_usage_report(self) -> Dict[str, Any]:
        """
        Export comprehensive usage report
        
        Returns:
            Dict containing detailed usage report
        """
        quota_status = self.get_quota_status()
        usage_history = self.get_usage_history(30)
        warnings = self.get_usage_warnings()
        
        return {
            "report_generated": datetime.now().isoformat(),
            "quota_status": quota_status,
            "usage_history_30_days": usage_history,
            "active_warnings": warnings,
            "api_statistics": self.data["google_gemini_api"]["usage_stats"],
            "tracking_metadata": self.data["tracking_metadata"]
        }


# Global instance for easy access
api_tracker = APIUsageTracker()


def get_api_tracker() -> APIUsageTracker:
    """
    Get the global API tracker instance
    
    Returns:
        APIUsageTracker instance
    """
    return api_tracker


def display_quota_status_sidebar() -> None:
    """
    Display quota status in Streamlit sidebar
    """
    tracker = get_api_tracker()
    quota_status = tracker.get_quota_status()
    warnings = tracker.get_usage_warnings()
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔑 API Quota Status")
        
        # Quota progress bar
        progress_value = quota_status["usage_percentage"] / 100
        st.progress(progress_value)
        
        # Status display
        st.markdown(f"**Status:** {quota_status['quota_status']}")
        st.markdown(f"**Used:** {quota_status['used_quota']}/{quota_status['total_quota']} ({quota_status['usage_percentage']}%)")
        st.markdown(f"**Remaining:** {quota_status['remaining_quota']} calls")
        st.markdown(f"**Today:** {quota_status['daily_usage']} calls")
        
        # Warnings
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        # Quick stats
        with st.expander("📊 Usage Details"):
            st.metric("Daily Usage", quota_status["daily_usage"])
            st.metric("Monthly Usage", quota_status["monthly_usage"])
            st.metric("Remaining", quota_status["remaining_quota"])


def check_api_quota_before_call() -> Tuple[bool, str]:
    """
    Check if API call can be made and return status
    
    Returns:
        Tuple of (can_call, message)
    """
    tracker = get_api_tracker()
    return tracker.can_make_api_call()