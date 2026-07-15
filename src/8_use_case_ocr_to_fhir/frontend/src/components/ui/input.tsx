import * as React from 'react';

import { cn } from '../../lib/utils';

interface InputProps extends React.ComponentProps<'input'> {
	variant?: 'flat' | 'default';
	hoverable?: boolean;
}

function Input({
	className,
	type,
	variant = 'default',
	hoverable,
	...props
}: InputProps) {
	return (
		<input
			type={type}
			data-slot='input'
			className={cn(
				// Base styles
				'file:text-foreground placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground flex h-9 w-full min-w-0 rounded-md px-3 py-1 text-base shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
				'bg-input border-border border',
				'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
				'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
				// Variant styles
				variant === 'default' && ['dark:bg-input/30'],
				variant === 'flat' && ['h-6'],
				// Hoverable styles
				hoverable && 'hover:bg-muted/50',
				className,
			)}
			{...props}
		/>
	);
}

export { Input };
