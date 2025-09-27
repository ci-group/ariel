export default {
    template: "",
    props: {
        options: Array,
    },
    data() {
    },
    mounted() {
        const TestNode = BaklavaJS.Core.defineNode({
            type: "TestNode",
            inputs: {
                a: () => new BaklavaJS.RendererVue.TextInputInterface("Hello", "world"),
                c: () => new BaklavaJS.RendererVue.ButtonInterface("Name", () => console.log("Button clicked")),
            },
            outputs: {
                b: () => new BaklavaJS.RendererVue.TextInputInterface("Hello", "world"),
            },
        });
        const MathNode = BaklavaJS.Core.defineNode({
            type: "MathNode",
            title: "Math",
            inputs: {
                operation: () =>
                    new BaklavaJS.RendererVue.SelectInterface("Operation", "Add", ["Add", "Subtract"]).setPort(
                        false
                    ),
                num1: () => new BaklavaJS.RendererVue.NumberInterface("Num 1", 1),
                num2: () => new BaklavaJS.RendererVue.NumberInterface("Num 2", 1)
            },
            outputs: {
                result: () => new BaklavaJS.Core.NodeInterface("Result")
            },
            calculate({ num1, num2, operation }) {
                console.log("Hey there")
                if (operation === "Add") {
                    return { result: num1 + num2 };
                } else {
                    return { result: num1 - num2 };
                }
            }
        });
        const DisplayNode = BaklavaJS.Core.defineNode({
            type: "DisplayNode",
            title: "Display",
            inputs: {
                value: () => new BaklavaJS.Core.NodeInterface("Value", "")
            },
            outputs: {
                display: () => new BaklavaJS.RendererVue.TextInterface("Display", "")
            },
            calculate({ value }) {
                return {
                    display: typeof value === "number" ? value.toFixed(3) : String(value)
                };
            }
        });
        viewModel.editor.registerNodeType(MathNode);
        viewModel.editor.registerNodeType(DisplayNode);
    },
    methods: {
    },
};