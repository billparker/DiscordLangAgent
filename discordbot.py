import os
import sys
import platform
import random
import asyncio
import json
import logging
import traceback

import discord
from discord.ext import commands, tasks
from discord.ext.commands import Bot, Context
from dotenv import load_dotenv
import aiosqlite
from helpers import db_manager
from langchain_community.llms import Ollama

if not os.path.isfile(f"{os.path.realpath(os.path.dirname(__file__))}/config.json"):
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open(f"{os.path.realpath(os.path.dirname(__file__))}/config.json") as file:
        config = json.load(file)

# Load .env file
load_dotenv()

# Initialize bot
intents = discord.Intents.all()
bot = Bot(command_prefix="/", intents=intents, help_command=None)

# Get environment variables
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
OWNERS = os.getenv("OWNERS")

bot.endpoint = os.getenv("OLLAMAENDPOINT")
if len(bot.endpoint.split("/api")) > 0:
    bot.endpoint = bot.endpoint.split("/api")[0]
bot.chatlog_dir = "chatlog_dir"
bot.endpoint_connected = False
bot.channel_list = [int(x) for x in CHANNEL_ID.split(",")]
bot.owners = [int(x) for x in OWNERS.split(",")]

class LoggingFormatter(logging.Formatter):
    # Colors
    black = "\x1b[30m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    gray = "\x1b[38m"
    # Styles
    reset = "\x1b[0m"
    bold = "\x1b[1m"

    COLORS = {
        logging.DEBUG: gray + bold,
        logging.INFO: blue + bold,
        logging.WARNING: yellow + bold,
        logging.ERROR: red,
        logging.CRITICAL: red + bold,
    }

    def format(self, record):
        log_color = self.COLORS[record.levelno]
        format = "(black){asctime}(reset) (levelcolor){levelname:<8}(reset) (green){name}(reset) {message}"
        format = format.replace("(black)", self.black + self.bold)
        format = format.replace("(reset)", self.reset)
        format = format.replace("(levelcolor)", log_color)
        format = format.replace("(green)", self.green + self.bold)
        formatter = logging.Formatter(format, "%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)

logger = logging.getLogger("discord_bot")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(LoggingFormatter())
# File handler
file_handler = logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
file_handler_formatter = logging.Formatter(
    "[{asctime}] [{levelname:<8}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{"
)
file_handler.setFormatter(file_handler_formatter)

# Add the handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
bot.logger = logger

# Set the llm attribute before loading extensions
bot.llm = Ollama(model="llama3")

async def init_db():
    async with aiosqlite.connect(
        f"{os.path.realpath(os.path.dirname(__file__))}/database/database.db"
    ) as db:
        with open(
            f"{os.path.realpath(os.path.dirname(__file__))}/database/schema.sql"
        ) as file:
            await db.executescript(file.read())
        await db.commit()

bot.config = config

@bot.event
async def on_ready():
    await db_manager.setup_db()
    bot.logger.info(f"Setting up database...")
    bot.logger.info(f"Logged in as {bot.user.name}")
    bot.logger.info(f"discord.py API version: {discord.__version__}")
    bot.logger.info(f"Python version: {platform.python_version()}")
    bot.logger.info(f"Running on: {platform.system()} {platform.release()} ({os.name})")
    bot.logger.info("-------------------")
    status_task.start()
    if config["sync_commands_globally"]:
        bot.logger.info("Syncing commands globally...")
        await bot.tree.sync()
    bot.logger.info(f"{bot.user.name} has connected to:")
    for items in bot.channel_list:
        try:
            channel = bot.get_channel(int(items))
            guild = channel.guild
            if isinstance(channel, discord.TextChannel):
                channel_name = channel.name
                bot.logger.info(f"{guild.name} \ {channel_name}")
            else:
                bot.logger.info(f"Channel with ID {items} is not a text channel")
        except AttributeError:
            bot.logger.info(
                "\n\n\n\nERROR: Unable to retrieve channel from .env \nPlease make sure you're using a valid channel ID, not a server ID."
            )

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the bot was mentioned in the message or if it's a direct message
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        bot.logger.info(f"Received message from {message.author.display_name}: {message.content}")
        
        if isinstance(message.channel, discord.DMChannel):
            response = await bot.get_cog("chatbot").chatbot.generate_response(message, message.content)
            await message.channel.send(response)
        else:
            await bot.process_commands(message)

@tasks.loop(minutes=6.0)
async def status_task() -> None:
    statuses = [
        "with Langchain🦜🔗",
    ]
    await bot.change_presence(activity=discord.Game(random.choice(statuses)))

@bot.event
async def on_command_completion(context: Context) -> None:
    full_command_name = context.command.qualified_name
    split = full_command_name.split(" ")
    executed_command = str(split[0])
    if context.guild is not None:
        bot.logger.info(
            f"Executed {executed_command} command in {context.guild.name} (ID: {context.guild.id}) by {context.author} (ID: {context.author.id})"
        )
    else:
        bot.logger.info(
            f"Executed {executed_command} command by {context.author} (ID: {context.author.id}) in DMs"
        )

@bot.event
async def on_command_error(context: Context, error) -> None:
    if isinstance(error, commands.CommandOnCooldown):
        minutes, seconds = divmod(error.retry_after, 60)
        hours, minutes = divmod(minutes, 60)
        hours = hours % 24
        embed = discord.Embed(
            description=f"**Please slow down** - You can use this command again in {f'{round(hours)} hours' if round(hours) > 0 else ''} {f'{round(minutes)} minutes' if round(minutes) > 0 else ''} {f'{round(seconds)} seconds' if round(seconds) > 0 else ''}.",
            color=0xE02B2B,
        )
        await context.send(embed=embed)
    elif isinstance(error, commands.CommandInvokeError):
        embed = discord.Embed(
            description="An error occurred while executing the command.",
            color=0xE02B2B
        )
        await context.send(embed=embed)
        bot.logger.error(f"Error in command {context.command}: {str(error)}")
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    elif isinstance(error, commands.CheckFailure):
        embed = discord.Embed(
            description="You do not have permission to use this command.",
            color=0xE02B2B
        )
        await context.send(embed=embed)
    elif isinstance(error, commands.MissingPermissions):
        embed = discord.Embed(
            description="You are missing the permission(s) `"
            +", ".join(error.missing_permissions)
            +"` to execute this command!",
            color=0xE02B2B,
        )
        await context.send(embed=embed)
    elif isinstance(error, commands.BotMissingPermissions):
        embed = discord.Embed(
            description="I am missing the permission(s) `"
            +", ".join(error.missing_permissions)
            +"` to fully perform this command!",
            color=0xE02B2B,
        )
        await context.send(embed=embed)
    elif isinstance(error, commands.MissingRequiredArgument):
        embed = discord.Embed(
            title="Error!",
            description=str(error).capitalize(),
            color=0xE02B2B,
        )
        await context.send(embed=embed)
    else:
        raise error

async def load_cogs() -> None:
    for file in os.listdir(f"{os.path.realpath(os.path.dirname(__file__))}/cogs"):
        if file.endswith(".py"):
            extension = file[:-3]
            try:
                await bot.load_extension(f"cogs.{extension}")
            except Exception as e:
                error_info = f"Failed to load extension {extension}. {type(e).__name__}: {e}"
                print(error_info)
                logging.error(f"Traceback: {traceback.format_exc()}")

asyncio.run(load_cogs())
asyncio.run(init_db())
bot.run(DISCORD_BOT_TOKEN)
