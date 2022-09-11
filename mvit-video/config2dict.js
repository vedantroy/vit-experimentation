#! /bin/env node

const fs = require("fs")

// From REV_VIT_S (I think?)
const FILE = "reversible_config.txt"

const text = fs.readFileSync(FILE).toString()
const lines = text.split("\n")

const newLines = []
let isFirst = true
for (const line of lines) {
    if (line.trim().length === 0) {
        continue
    }

    const isComment = line.trim().startsWith("#")
    if (isComment) {
        console.log(`Skipping: ${line}`)
        continue
    }

    let newLine = line.replace(":", "=")
    if (newLine[0] != " ") {
        newLine += "dict("
        if (!isFirst) {
            newLine = `),\n${newLine}`
        }
    } else {
        // quote strings
        const eqIdx = newLine.indexOf("=")
        let afterEq = newLine.slice(eqIdx + 1).trim()
        if (afterEq !== "False" && afterEq !== "True" && afterEq[0] !== "[" && isNaN(parseFloat(afterEq))) {
            afterEq = `"${afterEq}"`
        }
        newLine = newLine.slice(0, eqIdx + 1) + afterEq + ","
    }
    newLines.push(newLine)

    isFirst = false
}

newLines.push(")")
const newText = newLines.join("\n")
fs.writeFileSync("./config_dict.py", newText)
