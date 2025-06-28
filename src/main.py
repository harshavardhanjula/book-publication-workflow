"""Interactive workflow for scraping, rewriting, and refining content."""
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.ai.writer import AIWriter
from src.ai.reviewer import AIReviewer
from src.ai.rl_agent import RLContentAgent
from src.db import Database
from scraper.scraper import ChapterScraper
from src.config import settings

# Initialize console for rich output
console = Console()

class InteractiveWorkflow:
    """Interactive workflow for content scraping and refinement."""
    
    def __init__(self):
        """Initialize the workflow with required components."""
        self.db = Database()
        self.ai_writer = AIWriter()
        self.ai_reviewer = AIReviewer()
        self.rl_agent = RLContentAgent()
        self.scraper = ChapterScraper()
        self.current_content = ""
        self.current_metadata = {}
        self.current_state = "beginning"
        self.last_action_idx = None
        self.style_guide = {
            "tone": "professional",
            "audience": "general",
            "style_rules": [
                "Use clear and concise language",
                "Maintain a professional tone",
                "Ensure proper grammar and punctuation"
            ]
        }
        self.version_history = []  # Track all versions of the content
        self.current_version = 0  # Track current version number
        
        # Initialize saved IDs for database operations
        self.saved_book_id = None
        self.saved_chapter_id = None
    
    async def get_url_and_scrape(self) -> Dict:
        """Get URL from user and scrape content."""
        console.print("\n[bold blue]Content Scraper[/bold blue]")
        console.print("=" * 40)
        
        while True:
            url = input("\nEnter the URL of the content to process (or 'exit' to quit): ").strip()
            
            if url.lower() == 'exit':
                return {"success": False, "message": "Operation cancelled by user"}
            
            if not url.startswith(('http://', 'https://')):
                console.print("[red]Error:[/red] Please enter a valid URL starting with http:// or https://")
                continue
                
            try:
                console.print(f"\n[cyan]Scraping content from: {url}[/cyan]")
                scraped = await self.scraper.scrape_chapter(url)
                
                if not scraped or 'content' not in scraped or not scraped['content'].strip():
                    console.print("[red]Error:[/red] No content found at the provided URL")
                    continue
                    
                self.current_content = scraped['content']
                self.current_metadata = {
                    'source_url': url,
                    'title': scraped.get('title', 'Untitled'),
                    'screenshot_path': scraped.get('screenshot_path', '')
                }
                
                console.print("\n[green]✓ Content scraped successfully![/green]")
                if 'screenshot_path' in scraped and scraped['screenshot_path']:
                    console.print(f"Screenshot saved to: {scraped['screenshot_path']}")
                
                return {
                    'success': True,
                    'content': self.current_content,
                    'metadata': self.current_metadata
                }
                
            except Exception as e:
                logger.error(f"Error scraping content: {str(e)}")
                console.print(f"[red]Error:[/red] Failed to scrape content - {str(e)}")
                continue
    
    async def process_with_ai(self) -> Dict:
        """Show content and proceed to refinement options."""
        if not self.current_content:
            return {"success": False, "error": "No content to process"}
            
        console.print("\n[bold blue]Content Ready for Refinement[/bold blue]")
        console.print("=" * 40)
        
        # Add initial version to history
        self.version_history = [{
            'content': self.current_content,
            'version': 1,
            'description': 'Initial version'
        }]
        self.current_version = 1
        
        # Display extracted content
        console.print("\n[bold]Current Content (v1 - Initial version):[/bold]")
        console.print("-" * 50)
        console.print(Markdown(self.current_content[:1000] + ("..." if len(self.current_content) > 1000 else "")))
        console.print("-" * 50)
        
        # Proceed directly to interactive refinement
        console.print("\n[cyan]Content loaded. You can now refine it using the options below.[/cyan]")
        return await self.interactive_refinement()
    
    async def interactive_refinement(self) -> Dict:
        """Interactively refine the content with user feedback."""
        if not self.current_content:
            return {"success": False, "error": "No content to refine"}
            
        console.print("\n[bold blue]Interactive Refinement using AI[/bold blue]")
        console.print("=" * 40)
        
        iteration = 1
        
        while True:
            # Get RL suggestion
            rl_suggestion, action_idx = self.rl_agent.get_suggestion(self.current_content)
            self.last_action_idx = action_idx
            self.current_state = self.rl_agent._get_state(self.current_content)
            
            console.print("\n[bold blue]Refinement Options:[/bold blue]")
            console.print("- Type your instruction to refine (e.g., 'make it simple', 'make it sound cool')")
            console.print("- Type 'review' to get AI feedback")
            console.print("- Type 'preview' to view the current content")
            console.print("- Type 'save' to save the current version")
            console.print("- Type 'exit' to discard changes")
            
            user_input = input("\nEnter your choice: ").strip()
            
            # Convert input to lowercase for easier comparison
            user_input = user_input.lower()
            
            # Handle special commands
            if user_input in ['exit', 'quit']:
                return {"success": False, "error": "User exited without saving"}
                
            if user_input == 'preview':
                self._preview_content()
                continue
                
            if user_input == 'review':
                # Get AI review of current version
                with console.status("[bold green]Analyzing content with AI Reviewer..."):
                    review = await self.ai_reviewer.review_chapter(
                        content=self.current_content,
                        style_guide=self.style_guide,
                        chapter_title=self.current_metadata.get('title')
                    )
                
                if review['success']:
                    console.print("\n[bold blue]AI Review:[/bold blue]")
                    console.print("-" * 50)
                    console.print(f"Overall Score: {review['score']}/100")
                    console.print("\n[bold]Feedback:[/bold]")
                    console.print(review['feedback'])
                    if review['suggestions']:
                        console.print("\n[bold]Suggestions:[/bold]")
                        for i, suggestion in enumerate(review['suggestions'][:3], 1):
                            console.print(f"{i}. {suggestion}")
                    console.print("-" * 50)
                    input("\nPress Enter to continue...")
                else:
                    console.print(f"[red]Error getting review: {review.get('error', 'Unknown error')}[/red]")
                continue
               
            if user_input.lower() == 'save':
                # Give positive reward for saving
                if self.last_action_idx is not None:
                    self.rl_agent.update_model(
                        state=self.current_state,
                        action_idx=self.last_action_idx,
                        reward=1.0,  # Positive reward for saving
                        next_state='end'
                    )
                    self.rl_agent.save_model()
                return await self.save_to_database()
                
            
            # Process user input as refinement instruction
            if user_input and user_input not in ['save']:
                try:
                    console.print(f"\n[cyan]Processing: {user_input}[/cyan]")
                    
                    processed = await self.ai_writer.rewrite_chapter(
                        content=self.current_content,
                        style_guide=self.style_guide,
                        chapter_title=self.current_metadata.get('title', ''),
                        instruction=user_input
                    )
                    
                    if processed.get('success'):
                        # Update RL model with reward based on content change
                        if self.last_action_idx is not None:
                            # Simple reward: if content changed, give positive reward
                            content_changed = processed['content'] != self.current_content
                            reward = 1.0 if content_changed else -0.5
                            
                            self.rl_agent.update_model(
                                state=self.current_state,
                                action_idx=self.last_action_idx,
                                reward=reward,
                                next_state=self.rl_agent._get_state(processed['content'])
                            )
                            self.rl_agent.save_model()
                        
                        # Show version information before updating content
                        console.print("\n[bold blue]Version Update:[/bold blue]")
                        console.print(f"[dim]Updating to version {iteration + 1}...[/dim]")
                        
                        # Display the changes
                        console.print("\n[bold]Changes Made:[/bold]")
                        console.print("-" * 30)
                        console.print(Markdown(processed['content']))
                        console.print("-" * 30)
                        
                        # Update the content and version history
                        self.current_content = processed['content']
                        version_num = len(self.version_history) + 1
                        self.version_history.append({
                            'content': self.current_content,
                            'version': version_num,
                            'description': user_input[:100],  # Store first 100 chars of instruction as description
                            'timestamp': str(datetime.now())  # Add timestamp
                        })
                        self.current_version = version_num
                        console.print(f"[green]✓ Changes applied! (v{version_num})[/green]")
                        iteration += 1
                    else:
                        console.print(f"[yellow]No changes made: {processed.get('error', 'Unknown error')}[/yellow]")
                        
                        # Give negative reward for failed operation
                        if self.last_action_idx is not None:
                            self.rl_agent.update_model(
                                state=self.current_state,
                                action_idx=self.last_action_idx,
                                reward=-1.0,
                                next_state=self.current_state
                            )
                            self.rl_agent.save_model()
                        
                except Exception as e:
                    console.print(f"[red]Error during refinement: {str(e)}[/red]")
                    
                    # Give negative reward for errors
                    if self.last_action_idx is not None:
                        self.rl_agent.update_model(
                            state=self.current_state,
                            action_idx=self.last_action_idx,
                            reward=-1.0,
                            next_state=self.current_state
                        )
                        self.rl_agent.save_model()
    
    def _preview_content(self):
        """Helper method to preview the current content with version history."""
        while True:
            console.clear()
            console.print("\n[bold blue]Content Preview[/bold blue]")
            console.print("=" * 80)
            
            # Show version history
            console.print("\n[bold]Version History:[/bold]")
            for i, version in enumerate(self.version_history, 1):
                version_marker = "➤ " if version['version'] == self.current_version else "  "
                console.print(f"{version_marker}v{version['version']}: {version.get('description', 'No description')}")
            
            # Get the version to preview (default to current version)
            current_version_data = next((v for v in self.version_history if v['version'] == self.current_version), None)
            if not current_version_data:
                console.print("[red]Error: Current version not found in history[/red]")
                break
                
            content_to_show = current_version_data['content']
            
            # Display content preview
            console.print(f"\n[bold]Current Version: v{current_version_data['version']}[/bold]")
            if 'description' in current_version_data:
                console.print(f"[dim]{current_version_data['description']}[/dim]")
            console.print("-" * 80)
            
            # Show preview with limited length
            preview_length = 2000
            if len(content_to_show) > preview_length:
                last_space = content_to_show.rfind(' ', 0, preview_length)
                preview_text = content_to_show[:last_space] + "..."
                is_truncated = True
            else:
                preview_text = content_to_show
                is_truncated = False
                
            console.print(Markdown(preview_text))
            
            if is_truncated:
                console.print(f"\n[dim]Showing {len(preview_text)} of {len(content_to_show)} characters[/dim]")
            
            console.print("\n[dim]" + "-" * 80 + "[/dim]")
            
            # Navigation options
            console.print("\n[bold]Navigation:[/bold]")
            if len(self.version_history) > 1:
                console.print("- Enter version number to view (e.g., '1' for v1)")
            console.print("- 'full' - View full content of current version")
            if self.current_version != self.version_history[-1]['version']:
                console.print("- 'latest' - View latest version")
            console.print("- 'back' - Return to refinement")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == '':
                continue
            elif choice == 'back':
                break
            elif choice == 'full':
                console.clear()
                console.print(f"\n[bold]Full Content (v{current_version_data['version']}):[/bold]")
                if 'description' in current_version_data:
                    console.print(f"[dim]{current_version_data['description']}[/dim]")
                console.print("-" * 80)
                console.print(Markdown(content_to_show))
                console.print("\n[dim]" + "-" * 80 + "[/dim]")
                input("\nPress Enter to continue...")
            elif choice == 'latest' and self.current_version != self.version_history[-1]['version']:
                self.current_version = self.version_history[-1]['version']
            elif choice.isdigit():
                version_num = int(choice)
                if any(v['version'] == version_num for v in self.version_history):
                    self.current_version = version_num
                else:
                    console.print(f"[yellow]Version {version_num} not found. Showing current version.[/yellow]")
                    input("\nPress Enter to continue...")
            else:
                console.print("[yellow]Invalid option. Please try again.[/yellow]")
                input("\nPress Enter to continue...")
        
        console.clear()
    
    async def save_to_database(self) -> Dict:
        """Save the current content to the database with version history."""
        if not self.current_content:
            return {"success": False, "error": "No content to save"}
        
        console.print("\n[bold blue]Save Options[/bold blue]")
        console.print("=" * 40)
        
        try:
            # Generate a book ID if not exists
            title = self.current_metadata.get('title', 'Untitled')
            book_id = self.current_metadata.get('book_id') or f"book_{title.lower().replace(' ', '_')}"
            
            # Prepare version history for saving
            version_history_metadata = []
            for version in self.version_history:
                version_data = {
                    'version': version['version'],
                    'description': version.get('description', 'No description'),
                    'timestamp': version.get('timestamp', str(datetime.now())),
                    'content_length': len(version['content'])
                }
                version_history_metadata.append(version_data)
            
            # Prepare metadata with version information
            metadata = {
                'source_url': self.current_metadata.get('source_url', ''),
                'screenshot_path': self.current_metadata.get('screenshot_path', ''),
                'style_guide': self.style_guide,
                'total_versions': len(self.version_history),
                'current_version': self.current_version,
                'version_history': version_history_metadata,
                'last_updated': str(datetime.now())
            }
            
            # Show save summary
            console.print(f"\n[bold]Saving:[/bold] {title}")
            console.print(f"[dim]Versions: {len(self.version_history)}")
            console.print(f"Current version: v{self.current_version}")
            console.print(f"Total characters: {len(self.current_content)}")
            
            # Add to database
            chapter_id = self.db.add_chapter(
                book_id=book_id,
                title=title,
                content=self.current_content,
                chapter_number=1,  # Default to 1 for single chapter
                metadata=metadata
            )
            
            # Store the saved IDs for potential viewing
            self.saved_book_id = book_id
            self.saved_chapter_id = chapter_id
            
            console.print("\n[green]✓ Content saved successfully in ChromeDB![/green]")
            console.print(f"\n[bold]Book ID:[/bold] {book_id}")
            console.print(f"[bold]Chapter ID:[/bold] {chapter_id}")
            console.print(f"[bold]Saved version:[/bold] v{self.current_version}")
            
            # Ask if user wants to view the saved content
            while True:
                console.print("\n[bold]Options:[/bold]")
                console.print("1. View saved content")
                console.print("2. View version history")
                console.print("3. Return to main menu")
                
                choice = input("\nYour choice (1-3): ").strip().lower()
                
                if choice in ['1', 'view']:
                    console.clear()
                    console.print(f"\n[bold]Saved Content (v{self.current_version}):[/bold]")
                    if 'description' in self.version_history[-1]:
                        console.print(f"[dim]{self.version_history[-1]['description']}[/dim]")
                    console.print("-" * 80)
                    console.print(Markdown(self.current_content[:2000] + 
                                       ("..." if len(self.current_content) > 2000 else "")))
                    if len(self.current_content) > 2000:
                        console.print(f"\n[dim]Showing 2000 of {len(self.current_content)} characters. Use 'preview' to see more.[/dim]")
                    console.print("-" * 80)
                    input("\nPress Enter to continue...")
                    console.clear()
                elif choice in ['2', 'history']:
                    self._preview_content()
                elif choice in ['3', 'exit', 'back', '']:
                    console.print("\n[bold]Returning to main menu...[/bold]")
                    break
                else:
                    console.print("[yellow]Invalid option. Please try again.[/yellow]")
            
            return {
                'success': True,
                'book_id': book_id,
                'chapter_id': chapter_id,
                'version': self.current_version,
                'total_versions': len(self.version_history)
            }
            
        except Exception as e:
            error_msg = f"Error saving to database: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]Error:[/red] {error_msg}")
            return {"success": False, "error": error_msg}
    
    async def run(self):
        """Run the interactive workflow."""
        console.print("\n[bold blue]Automated Book Publication Workflow[/bold blue]")
        console.print("=" * 40)
        
        # Step 1: Get URL and scrape content
        result = await self.get_url_and_scrape()
        if not result['success']:
            return result
        
        # Step 2: Process with AI
        result = await self.process_with_ai()
        if not result['success']:
            return result
        
        # Step 3: Interactive refinement
        result = await self.interactive_refinement()
        if not result['success']:
            return result
        
        # Step 4: Save to database
        result = await self.save_to_database()
        return result


async def main():
    """Run the interactive workflow."""
    workflow = InteractiveWorkflow()
    await workflow.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user.[/red]")
    except Exception as e:
        console.print(f"\n[red]An error occurred: {str(e)}[/red]")
        logger.error(f"Workflow error: {str(e)}", exc_info=True)
    finally:
        console.print("\n[blue]Workflow completed. Goodbye![/blue]")
