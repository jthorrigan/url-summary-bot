"""
Command-line interface for the URL Summary Bot.
"""

import argparse
import os
from url_summary_bot import URLSummaryBot


def main():
    parser = argparse.ArgumentParser(
        description="Summarize URLs using FREE Hugging Face API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "https://example.com/article"
  python cli.py "https://example.com" --length short
  python cli.py "https://example.com" --style-file my_style.txt
  python cli.py "https://example.com" --instructions "Focus on technical details"
        """
    )
    parser.add_argument("url", help="URL to summarize")
    parser.add_argument(
        "--length",
        choices=["short", "medium", "long"],
        default="medium",
        help="Summary length (default: medium)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Custom maximum length in tokens"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        help="Custom minimum length in tokens"
    )
    parser.add_argument(
        "--style-file",
        help="Path to file containing style example"
    )
    parser.add_argument(
        "--style-url",
        help="URL containing style example"
    )
    parser.add_argument(
        "--instructions",
        help="Additional custom instructions"
    )
    parser.add_argument(
        "--model",
        default="facebook/bart-large-cnn",
        help="Hugging Face model to use (default: facebook/bart-large-cnn)"
    )
    parser.add_argument(
        "--no-jina",
        action="store_true",
        help="Don't use Jina AI for content extraction"
    )
    
    args = parser.parse_args()
    
    # Get API key from environment
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_api_key:
        print("⚠️  Warning: HUGGINGFACE_API_KEY not set.")
        print("Get your free API key at: https://huggingface.co/settings/tokens")
        print("Then run: export HUGGINGFACE_API_KEY='your-key-here'\n")
        return
    
    # Initialize bot
    bot = URLSummaryBot(hf_api_key)
    bot.set_model(args.model)
    
    # Load style example if provided
    style_example = None
    if args.style_file:
        with open(args.style_file, 'r') as f:
            style_example = f.read()
    elif args.style_url:
        print(f"Loading style from {args.style_url}...")
        style_example = bot.fetch_url_content(args.style_url, use_jina=not args.no_jina)
    
    # Generate summary
    result = bot.generate_summary(
        url=args.url,
        length=args.length,
        max_length=args.max_length,
        min_length=args.min_length,
        style_example=style_example,
        custom_instructions=args.instructions,
        use_jina=not args.no_jina
    )
    
    # Display result
    if result["success"]:
        print(f"\n{'='*70}")
        print(f"📄 URL: {result['url']}")
        print(f"📏 Length: {result['length_target']}")
        print(f"🤖 Model: {result['model']}")
        print(f"🎨 Style applied: {result['style_applied']}")
        print(f"{'='*70}\n")
        print(result["summary"])
        print(f"\n{'='*70}")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()
