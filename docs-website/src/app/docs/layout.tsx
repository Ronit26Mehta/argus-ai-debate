import { DocsSidebar } from "@/components/layout/DocsSidebar"

export default function DocsLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <div className="flex">
            <DocsSidebar />
            <div className="flex-1">
                <div className="container max-w-4xl py-10">
                    {children}
                </div>
            </div>
        </div>
    )
}
