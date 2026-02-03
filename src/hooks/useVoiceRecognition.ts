import { useEffect, useRef, useCallback, useState } from 'react';
import { useSandbox } from '@/context/SandboxContext';
import type { VoiceCommand } from '@/types';
import { parseColorName, parseShapeType, getColorHex } from '@/constants/shapes';

// Extend Window interface for webkit speech recognition
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}

declare global {
  interface Window {
    webkitSpeechRecognition: new () => SpeechRecognition;
    SpeechRecognition: new () => SpeechRecognition;
  }
}

interface UseVoiceRecognitionOptions {
  onCommand?: (command: VoiceCommand) => void;
  language?: string;
}

export function useVoiceRecognition(options?: UseVoiceRecognitionOptions) {
  const { state, setVoice, addObject, removeObject, clearAllObjects } = useSandbox();
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [isSupported, setIsSupported] = useState(false);

  // Parse voice command
  const parseCommand = useCallback((transcript: string): VoiceCommand | null => {
    const normalized = transcript.toLowerCase().trim();
    
    // Create command: "create [color] [shape]"
    if (normalized.includes('create') || normalized.includes('add') || normalized.includes('make')) {
      const shape = parseShapeType(normalized);
      if (shape) {
        const color = parseColorName(normalized);
        return {
          type: 'create',
          shape,
          color: color || undefined,
        };
      }
    }

    // Delete command
    if (normalized.includes('delete') || normalized.includes('remove')) {
      return { type: 'delete' };
    }

    // Clear all command
    if (normalized.includes('clear all') || normalized.includes('delete all') || normalized.includes('remove all')) {
      return { type: 'clear' };
    }

    return null;
  }, []);

  // Execute voice command
  const executeCommand = useCallback((command: VoiceCommand) => {
    switch (command.type) {
      case 'create':
        if (command.shape && state.gesture.pointerPosition) {
          const color = command.color ? getColorHex(command.color) : undefined;
          addObject(command.shape, state.gesture.pointerPosition, color);
        }
        break;

      case 'delete':
        if (state.selectedObjectId) {
          removeObject(state.selectedObjectId);
        }
        break;

      case 'clear':
        clearAllObjects();
        break;
    }

    options?.onCommand?.(command);
  }, [
    state.gesture.pointerPosition,
    state.selectedObjectId,
    addObject,
    removeObject,
    clearAllObjects,
    options,
  ]);

  // Initialize speech recognition
  useEffect(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognitionAPI) {
      setIsSupported(false);
      setVoice({ isSupported: false, error: 'Speech recognition not supported' });
      return;
    }

    setIsSupported(true);
    setVoice({ isSupported: true });

    const recognition = new SpeechRecognitionAPI();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = options?.language || 'en-US';

    recognition.onstart = () => {
      setVoice({ isListening: true, error: null });
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const lastResult = event.results[event.results.length - 1];
      const transcript = lastResult[0].transcript;
      
      setVoice({ lastCommand: transcript });

      // Only process final results
      if (lastResult.isFinal) {
        const command = parseCommand(transcript);
        if (command) {
          setVoice({ lastParsedCommand: command });
          executeCommand(command);
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error('Speech recognition error:', event.error);
      setVoice({ error: event.error, isListening: false });
    };

    recognition.onend = () => {
      setVoice({ isListening: false });
      
      // Auto-restart if still mounted
      if (recognitionRef.current) {
        try {
          recognition.start();
        } catch (e) {
          // Ignore - may already be running
        }
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
        recognitionRef.current = null;
      }
    };
  }, [options?.language, parseCommand, executeCommand, setVoice]);

  // Start listening
  const startListening = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.start();
      } catch (e) {
        // May already be running
      }
    }
  }, []);

  // Stop listening
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  }, []);

  return {
    isSupported,
    isListening: state.voice.isListening,
    lastCommand: state.voice.lastCommand,
    lastParsedCommand: state.voice.lastParsedCommand,
    error: state.voice.error,
    startListening,
    stopListening,
  };
}

export default useVoiceRecognition;
