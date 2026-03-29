import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.config import settings
import logging

logger = logging.getLogger("MedSarthi-Email")

def send_email_async(subject: str, recipient: str, html_content: str):
    """
    Sends a real email using SMTP. This should be wrapped in FastAPI's BackgroundTasks.
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(f"SMTP Credentials not set. Logging email to console for development.")
        logger.info(f"TO: {recipient} | SUBJECT: {subject}\n{html_content}")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = f"{settings.EMAILS_FROM_NAME} <{settings.EMAILS_FROM_EMAIL or settings.SMTP_USER}>"
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {recipient}")
    except Exception as e:
        logger.error(f"Failed to send email to {recipient}: {str(e)}")

def get_html_template(title, body_html, footer_text=""):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8fafc; margin: 0; padding: 0; color: #334155; }}
            .container {{ max-width: 600px; margin: 40px auto; background-color: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; }}
            .header {{ background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%); padding: 40px 20px; text-align: center; color: white; }}
            .header h1 {{ margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.025em; }}
            .content {{ padding: 40px; line-height: 1.6; font-size: 16px; }}
            .button {{ display: inline-block; background-color: #4f46e5; color: #ffffff !important; padding: 14px 28px; border-radius: 12px; text-decoration: none; font-weight: 700; margin-top: 24px; box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2); transition: transform 0.2s; }}
            .footer {{ background-color: #f1f5f9; padding: 20px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
            .badge {{ display: inline-block; background-color: #f1f5f9; color: #64748b; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 700; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MedSarthi</h1>
                <p style="margin-top: 8px; opacity: 0.9;">AI-First Clinical Intelligence</p>
            </div>
            <div class="content">
                <div class="badge">{title}</div>
                {body_html}
            </div>
            <div class="footer">
                <p>This is an automated notification from your MedSarthi portal.</p>
                <p>&copy; 2026 MedSarthi Healthcare Solutions. All rights reserved.</p>
                <p style="margin-top: 8px;">{footer_text}</p>
            </div>
        </div>
    </body>
    </html>
    """

def send_registration_email(recipient: str, username: str, temp_password: str):
    body = f"""
    <h2>Welcome to MedSarthi, {username}!</h2>
    <p>Your doctor has created a medical profile for you on the MedSarthi platform. You can now access your prescriptions, reports, and vital trends from our dashboard.</p>
    
    <div style="background-color: #f8fafc; padding: 24px; border-radius: 16px; margin: 24px 0; border: 1px solid #e2e8f0;">
        <p style="margin: 0; font-size: 12px; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Temporary Credentials</p>
        <p style="margin: 8px 0; font-weight: 700;">Username: <strong>{username}</strong></p>
        <p style="margin: 0; font-weight: 700;">Password: <code style="background-color: #4f46e5; color: white; padding: 2px 6px; border-radius: 4px;">{temp_password}</code></p>
    </div>
    
    <p>Please log in and update your password immediately to ensure account security.</p>
    <a href="http://localhost:5173/login" class="button">Log In to Dashboard</a>
    """
    html = get_html_template("New Patient Account", body)
    send_email_async("Welcome to MedSarthi - Your Credentials Inside", recipient, html)

def send_password_reset_email(recipient: str, token: str):
    reset_url = f"http://localhost:5173/reset-password?token={token}"
    body = f"""
    <h2>Password Reset Request</h2>
    <p>We received a request to reset your MedSarthi account password. If you didn't make this request, you can safely ignore this email.</p>
    <p>This secure link will expire in <strong>1 hour</strong>.</p>
    <a href="{reset_url}" class="button">Reset My Password</a>
    """
    html = get_html_template("Security Notification", body)
    send_email_async("Reset Your MedSarthi Password", recipient, html)

def send_appointment_notification(recipient: str, date: str, doctor_name: str):
    body = f"""
    <h2>Appointment Confirmed</h2>
    <p>Your appointment has been successfully scheduled. Here are the details:</p>
    <div style="background-color: #f0fdfa; padding: 24px; border-radius: 16px; margin: 24px 0; border: 1px solid #ccfbf1;">
        <p style="margin: 0; font-weight: 700; color: #0d9488;">Date & Time: {date}</p>
        <p style="margin: 4px 0; font-weight: 700; color: #0d9488;">Doctor: Dr. {doctor_name}</p>
    </div>
    <p>Please arrive 10 minutes before your scheduled time.</p>
    <a href="http://localhost:5173/patient/dashboard/appointments" class="button">View Appointment</a>
    """
    html = get_html_template("Appointment Update", body)
    send_email_async("Confirmed: Appointment with Dr. " + doctor_name, recipient, html)

def send_lab_order_notification(recipient: str, doctor_name: str, lab_tests: list):
    tests_html = "<ul>" + "".join([f"<li><strong>{test['name']}</strong></li>" for test in lab_tests]) + "</ul>"
    body = f"""
    <h2>New Diagnostic Order</h2>
    <p>Dr. {doctor_name} has issued a new lab investigation for you. Please visit your nearest diagnostic center to complete these tests.</p>
    
    <div style="background-color: #f1f5f9; padding: 20px; border-radius: 16px; margin: 24px 0;">
        <p style="font-weight: 800; font-size: 12px; color: #64748b; margin-bottom: 12px; text-transform: uppercase;">Tests Prescribed:</p>
        {tests_html}
    </div>
    
    <p>You can view and print the official lab order from your portal.</p>
    <a href="http://localhost:5173/patient/dashboard/history" class="button">View Lab Order</a>
    """
    html = get_html_template("Prescription Update", body)
    send_email_async("Action Required: New Lab Investigation Order", recipient, html)
