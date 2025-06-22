import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER")
        self.email_password = "dxkd hlqz zudh zfxj"
    
    async def send_confirmation_email(self, name: str, email: str, interview_date: str, interview_time: str, booking_time:str):
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = email
            msg['Subject'] = "Interview Confirmation"
            
            body = f"""
            Dear {name},
            
            Your interview has been successfully booked for:
            Date: {interview_date}
            Time: {interview_time}
            
            We look forward to meeting with you.
            
            Best regards,
            Suraj Acharya
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email send failed: {e}")
            return False