import requests
import urllib.parse
from loguru import logger
from config.env import get_env

def send_whatsapp_message(message: str) -> None:
    """
    Sends a WhatsApp message using CallMeBot API.

    Args:
        message (str): The message to send.

    Raises:
        ValueError: If required environment variables are not set.
    """
    try:
        env = get_env()

        # Ensure environment variables are set
        if not message:
            raise ValueError("Message cannot be empty.")

        # Fetch phone number and API key from environment variables
        if not env.PHONE_NUMBER or not env.CMB_API_KEY:
            logger.warning("Environment variables PHONE_NUMBER and CMB_API_KEY must be set to use the notification system.")
            return

        phone_number = env.PHONE_NUMBER
        api_key = env.CMB_API_KEY

        # Construct the API URL
        url = f"https://api.callmebot.com/whatsapp.php?phone={phone_number}&text={urllib.parse.quote(message)}&apikey={api_key}"

        response = requests.get(url)

        if response.status_code == 200:
            logger.info("Message sent successfully via WhatsApp.")
        else:
            logger.error(f"Error al enviar el mensaje: {response}")
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {e}")
        