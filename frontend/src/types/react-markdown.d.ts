declare module 'react-markdown' {
  import { ReactNode, ComponentType, HTMLProps } from 'react'

  export interface Components {
    [key: string]: ComponentType<any>
  }

  export interface ReactMarkdownOptions {
    children: string
    remarkPlugins?: any[]
    components?: {
      ol?: ComponentType<HTMLProps<HTMLOListElement>>
      li?: ComponentType<HTMLProps<HTMLLIElement>>
      p?: ComponentType<HTMLProps<HTMLParagraphElement>>
      // Add other HTML elements as needed
    }
    className?: string
  }

  const ReactMarkdown: (props: ReactMarkdownOptions) => ReactNode
  export default ReactMarkdown
}