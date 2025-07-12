variable "number" {
  type = number
}

variable "string" {
  type = string
}

variable "string_nullable" {
  type     = string
  nullable = true
  default  = "nullable_string"
}

variable "boolean" {
  type = bool
}

variable "list" {
  type = list(string)
}

variable "tuple" {
  type = tuple([string, number, bool])
}


variable "set" {
  type = set(string)
}


variable "map" {
  type = map(string)
}


variable "object" {
  type = object({
    field1 = string
    field2 = optional(string)
  })
}
