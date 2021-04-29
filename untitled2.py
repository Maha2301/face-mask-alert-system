import smtplib
from email.message import EmailMessage

def email_alert(subject, body, to):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to
    
    user = "mahalakshmithirumurthy@gmail.com"
    password = "qvlhwmdprzztfrpz"
    msg['from'] = user
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user,password)
    server.send_message(msg)
    server.quit()
    
if __name__ == '__main__':
    email_alert("Alert!!", "A person has violated facial mask policy. Kindly check the camera to recognise the person", "maha.lakshmi140@gmail.com")