import os
import argparse
import whisper
from pathlib import Path
from typing import Union, List
from numpy.typing import NDArray


def transcribe_audio_files(directory: str, model_size: str = 'base', 
                          file_extensions: List[str] = None, 
                          output_suffix: str = '', 
                          language: str = None) -> None:
    """
    Transcribe all audio/video files in a directory using OpenAI Whisper.
    
    Args:
        directory: Path to directory containing audio/video files
        model_size: Whisper model size (tiny, base, small, medium, large)
        file_extensions: List of file extensions to process
        output_suffix: Suffix to add to transcript filenames
        language: Language code to force (e.g., 'en' for English, None for auto-detect)
    """
    if file_extensions is None:
        file_extensions = ['.mp4', '.wav', '.mp3', '.m4a', '.avi', '.mov']
    
    directory = Path(directory)
    if not directory.exists():
        print(f'Error: Directory {directory} does not exist')
        return
    
    print(f'Loading Whisper model: {model_size}')
    model = whisper.load_model(model_size)
    
    # Find all audio/video files
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(directory.glob(f'*{ext}'))
        audio_files.extend(directory.glob(f'*{ext.upper()}'))
    
    if not audio_files:
        print(f'No audio/video files found with extensions: {file_extensions}')
        return
    
    print(f'Found {len(audio_files)} files to process')
    
    for i, file_path in enumerate(audio_files, 1):
        print(f'Processing ({i}/{len(audio_files)}): {file_path.name}')
        
        try:
            # Transcribe the audio
            if language:
                result = model.transcribe(str(file_path), language=language)
            else:
                result = model.transcribe(str(file_path))
            
            # Create output filename
            transcript_filename = f'{file_path.stem}{output_suffix}.txt'
            transcript_path = directory / transcript_filename
            
            # Output transcript to console
            print(f'  -> Transcript: {result["text"]}')
            
            # Save transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            
            print(f'  -> Transcript saved: {transcript_filename}')
            
        except Exception as e:
            print(f'  -> Error processing {file_path.name}: {str(e)}')
            continue


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe audio/video files using OpenAI Whisper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_audio_whisper.py /path/to/videos
  python transcribe_audio_whisper.py /path/to/videos --model_size small
  python transcribe_audio_whisper.py /path/to/videos --extensions .mp4 .wav
  python transcribe_audio_whisper.py /path/to/videos --language en --model_size base
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory containing audio/video files to transcribe'
    )
    
    parser.add_argument(
        '--model_size',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.mp4', '.wav', '.mp3', '.m4a', '.avi', '.mov'],
        help='File extensions to process (default: .mp4 .wav .mp3 .m4a .avi .mov)'
    )
    
    parser.add_argument(
        '--output_suffix',
        default='',
        help='Suffix to add to transcript filenames (default: "")'
    )
    
    parser.add_argument(
        '--language',
        help='Force specific language (e.g., "en" for English). If not specified, language will be auto-detected.'
    )
    
    args = parser.parse_args()
    
    transcribe_audio_files(
        directory=args.directory,
        model_size=args.model_size,
        file_extensions=args.extensions,
        output_suffix=args.output_suffix,
        language=args.language
    )


if __name__ == '__main__':
    main()
