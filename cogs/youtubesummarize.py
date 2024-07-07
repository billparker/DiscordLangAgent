import discord
from discord import app_commands
from discord.ext import commands
import langchain

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

import torch

class YoutubeSummaryCog(commands.Cog):

    def __init__(self, bot):
        self.bot = bot
        self.llm = self.bot.llm

    @app_commands.command(name="youtubesummary", description="Summarize a YouTube video given its URL")
    async def summarize(self, interaction: discord.Interaction, url: str):
        await interaction.response.defer()

        await interaction.followup.send(
            embed=discord.Embed(
                title=f"{interaction.user.display_name} used Youtube Summary 📺",
                description=f"Summarizing {url} \nGenerating response\nPlease wait..",
                color=0x9C84EF
            )
        )
        try:
            self.bot.logger.info(f"Loading transcript for URL: {url}")
            loader = YoutubeLoader.from_youtube_url(url)
            transcript = loader.load()

            self.bot.logger.info("Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
            texts = text_splitter.split_documents(transcript)

            self.bot.logger.info("Creating and configuring summarize chain...")
            chain = load_summarize_chain(llm=self.llm, chain_type="map_reduce", verbose=True)

            self.bot.logger.info("Running summarize chain...")
            summary = chain.run(texts)

            self.bot.logger.info(f"Summary generated: {summary}")
            await interaction.followup.send(f'Summary:\n{summary}')

        except Exception as e:
            self.bot.logger.error(f"Error occurred: {str(e)}")
            await interaction.channel.send(f'Sorry, an error occurred: {str(e)}')

async def setup(bot):
    await bot.add_cog(YoutubeSummaryCog(bot))
