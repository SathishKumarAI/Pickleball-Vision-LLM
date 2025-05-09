import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

class AlertManager:
    """
    A utility class for logging, error handling, and sending alerts.
    """

    def __init__(self, log_file: str = "pipeline.log", email_alerts: bool = False, email_config: Optional[dict] = None):
        """
        Initialize the AlertManager.

        Args:
            log_file (str): Path to the log file.
            email_alerts (bool): Whether to enable email alerts.
            email_config (Optional[dict]): Configuration for email alerts. Should include keys:
                - 'smtp_server': SMTP server address
                - 'smtp_port': SMTP server port
                - 'sender_email': Sender's email address
                - 'receiver_email': Receiver's email address
                - 'password': Sender's email password
        """
        self.logger = self._setup_logger(log_file)
        self.email_alerts = email_alerts
        self.email_config = email_config

    def _setup_logger(self, log_file: str) -> logging.Logger:
        """
        Set up the logger.

        Args:
            log_file (str): Path to the log file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger("PipelineLogger")
        logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_info(self, message: str):
        """
        Log an informational message.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)

    def log_error(self, message: str):
        """
        Log an error message.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)

    def send_email_alert(self, subject: str, body: str):
        """
        Send an email alert.

        Args:
            subject (str): Subject of the email.
            body (str): Body of the email.
        """
        if not self.email_alerts or not self.email_config:
            self.logger.warning("Email alerts are disabled or email configuration is missing.")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config["sender_email"]
            msg["To"] = self.email_config["receiver_email"]
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"]) as server:
                server.starttls()
                server.login(self.email_config["sender_email"], self.email_config["password"])
                server.send_message(msg)

            self.logger.info(f"Email alert sent: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def handle_exception(self, exception: Exception, context: str = ""):
        """
        Handle an exception by logging it and optionally sending an alert.

        Args:
            exception (Exception): The exception to handle.
            context (str): Additional context about where the exception occurred.
        """
        error_message = f"Exception occurred in {context}: {str(exception)}"
        self.log_error(error_message)

        if self.email_alerts:
            self.send_email_alert(
                subject="Critical Error in Pickleball Vision Pipeline",
                body=error_message
            )